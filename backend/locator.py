"""Two-pass, grid-based visual element locator.

Iterative grid refinement: the vision model is shown the full screenshot with
a coarse 10x10 grid and returns the cell range covering the target (every cell
the element touches, since elements often span several cells). That range
(plus margin) is then cropped from the full-resolution screenshot, upscaled,
re-gridded, and asked again — repeating until a grid row is smaller than the
click precision threshold. The click lands on the center of the final range
rectangle, which handles small boundary-straddling targets (radio buttons) and
wide ones (full-width input fields) alike.

Vision calls go through the LiveKit inference gateway, so only the LiveKit
credentials from .env are required (same as the conversational agent).

Set LOCATOR_DEBUG=1 to save the annotated images of the last locate call to
the debug/ directory.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from livekit.agents import APIConnectOptions, inference
from livekit.agents.llm import ChatContext, ImageContent
from PIL import Image

import grid

logger = logging.getLogger("element-locator")

# Use a strong vision model for grounding; the conversational LLM can stay small.
DEFAULT_MODEL = "openai/gpt-4.1"
MAX_UPLOAD_WIDTH = 1600
_CONN_OPTIONS = APIConnectOptions(timeout=30.0)

# Stop zooming once a grid row is this short (in screenshot pixels): the click
# quantization error is then at most half of this, well inside small controls
# like radio buttons (~40 screenshot px on Retina).
PRECISION_PX = 30
# Total grid passes including the initial coarse one.
MAX_PASSES = 4

_CELL_PROMPT = """You are looking at a screenshot with a red grid overlay. \
Each cell is labeled in its top-left corner (A1, B3, ...).

Target element: {description}

Elements often span several grid cells. Give the full extent of the target as
a cell range from its top-left cell to its bottom-right cell, for example
"B4:B8" for a wide input field, or a single cell like "C5" if it fits in one.
Include EVERY cell the target touches, even partially: a small control such as
a radio button that straddles a cell boundary must be reported as both cells,
for example "D3:E3", never just one of them.
If the target is a form control (input field, dropdown, button, radio button)
that has a separate text label, give the cells of the clickable control
itself, NOT the label text.

Respond with ONLY this JSON, no other text:
{{"found": true, "cells": "<top-left cell>:<bottom-right cell>"}}
If the target is not visible, respond with {{"found": false, "cells": null}}."""

_VERIFY_PROMPT = """You are looking at a screenshot of an application under test.

Expectation: {expectation}

Respond with ONLY this JSON, no other text:
{{"pass": true or false, "reason": "<one short sentence>"}}"""


def _image_to_data_url(image: Image.Image) -> str:
    if image.width > MAX_UPLOAD_WIDTH:
        ratio = MAX_UPLOAD_WIDTH / image.width
        image = image.resize((MAX_UPLOAD_WIDTH, int(image.height * ratio)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _parse_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class ElementLocator:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.llm = inference.LLM(model=model)
        self.debug_dir = Path(__file__).parent / "debug" if os.getenv("LOCATOR_DEBUG") else None

    async def locate(self, description: str) -> Optional[Tuple[float, float]]:
        """Find an element on the current screen by natural-language description.

        Returns (x, y) in logical screen points ready for pyautogui, or None.
        """
        screenshot, gridded, geometry = await asyncio.to_thread(grid.capture_with_grid)
        self._save_debug(gridded, "pass1")

        screenshot.save('my_grid-new.png', format='PNG')
        print("locate fn called")

        prompt = _CELL_PROMPT.format(description=description)
        cells = self._cells_from(await self._ask(prompt, gridded))
        if cells is None:
            logger.info(f"Locate {description!r}: not found on screen")
            return None
        target = geometry.range_center_logical(cells)
        if target is None:
            logger.warning(f"Locate {description!r}: invalid cells {cells!r}")
            return None

        # Keep zooming into the reported cells until the grid rows are smaller
        # than the click precision we need. Each pass crops the chosen range
        # (plus margin) from the full-resolution screenshot and re-grids it, so
        # a boundary-straddling radio button ends up well inside a cell within
        # a pass or two. Any failure along the way keeps the last good target.
        for pass_num in range(2, MAX_PASSES + 1):
            if geometry.cell_height * geometry.downscale <= PRECISION_PX:
                break
            fine = grid.crop_with_fine_grid(screenshot, geometry, cells)
            if fine is None:
                break
            crop, fine_geometry = fine
            self._save_debug(crop, f"pass{pass_num}")
            fine_cells = self._cells_from(await self._ask(prompt, crop))
            if fine_cells is None:
                logger.info(f"Locate {description!r}: pass {pass_num} lost the target, "
                            f"keeping previous result")
                break
            fine_target = fine_geometry.range_center_logical(fine_cells)
            if fine_target is None:
                break
            geometry, cells, target = fine_geometry, fine_cells, fine_target

        logger.info(
            f"Located {description!r}: final cells {cells} -> logical "
            f"({int(target[0])}, {int(target[1])})"
        )
        return target

    @staticmethod
    def _cells_from(data: Optional[dict]) -> Optional[str]:
        """Extract the cell range from a model response, tolerating either
        the 'cells' or legacy 'cell' attribute name."""
        if not data or not data.get("found"):
            return None
        cells = data.get("cells") or data.get("cell")
        return str(cells) if cells else None

    async def verify(self, expectation: str) -> Tuple[bool, str]:
        """Check an expectation against the current screen. Returns (passed, reason)."""
        screenshot, _ = await asyncio.to_thread(grid.capture_screenshot)
        data = await self._ask(_VERIFY_PROMPT.format(expectation=expectation), screenshot)
        if not data:
            return False, "could not parse verification response"
        return bool(data.get("pass")), str(data.get("reason", ""))

    async def _ask(self, prompt: str, image: Image.Image) -> Optional[dict]:
        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(
            role="user",
            content=[
                prompt,
                ImageContent(image=_image_to_data_url(image), inference_detail="high"),
            ],
        )
        text = ""
        stream = self.llm.chat(chat_ctx=chat_ctx, conn_options=_CONN_OPTIONS)
        async with stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
        data = _parse_json(text)
        if data is None:
            logger.warning(f"Could not parse model response: {text!r}")
        return data

    def _save_debug(self, image: Image.Image, name: str) -> None:
        if self.debug_dir is None:
            return
        self.debug_dir.mkdir(exist_ok=True)
        image.save(self.debug_dir / f"last_{name}.png", format="PNG")

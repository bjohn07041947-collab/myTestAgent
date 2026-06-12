"""OmniParser-based element locator (primary backend, grid as fallback).

Uses the OmniParser v2 icon-detection model (YOLO) to find every interactable
element on screen with a pixel-accurate bounding box, then asks the vision
model to pick the matching element from a numbered overlay (set-of-mark
prompting). One LLM call per locate instead of the grid locator's 2-4 zoom
passes, and clicks land on real element centers, so there is no grid
quantization error at all.

The model weights (microsoft/OmniParser-v2.0, icon_detect) are downloaded
from Hugging Face on first use and cached. 

This module imports torch/ultralytics at import time; when those optional
dependencies are missing, importing fails and ElementLocator silently sticks
to the grid backend.
"""

import asyncio
import logging
import os
import re
import threading
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Tuple

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import grid as grid_mod

logger = logging.getLogger("omni-locator")

WEIGHTS_REPO = "microsoft/OmniParser-v2.0"
WEIGHTS_FILE = "icon_detect/model.pt"

# OmniParser's recommended thresholds for the icon detector.
BOX_CONFIDENCE = 0.05
IOU_THRESHOLD = 0.1
# Inference resolution; full Retina screenshots need more than the YOLO
# default of 640 to catch small controls like radio buttons.
IMAGE_SIZE = 1280
MAX_ELEMENTS = 200

_SOM_PROMPT = """You are looking at a screenshot where interactable UI elements \
are outlined with red boxes, each tagged with a number at its top-left corner.

Target element: {description}

Pick the numbered box that best matches the target. If the target is a form
control (input field, dropdown, button, radio button) with a separate text
label, pick the box around the clickable control itself when both are boxed.
Some elements have no tight box of their own — bare input fields shown as
just an underline or a "$" often don't. In that case pick the SMALLEST
numbered box that CONTAINS the target (a section or panel box is fine; a
precise second pass will pinpoint the target inside it). NEVER substitute a
different element, such as a nearby link or help text, just because it has
its own box.

Match by meaning, not exact wording: the element may show an abbreviation,
code, or different phrasing than the description (a state dropdown may show
"FL" for Florida, a credit option may read "Excellent (740 and above)" for
excellent). The abbreviation or rephrasing must expand to EXACTLY the
described item: "FL" matches Florida, but "FL" does NOT match Texas — a
different item of the same kind is NOT a match, even if it is the only one
of its kind on screen.

Respond with ONLY this JSON, no other text:
{{"found": true, "id": <box number>, "evidence": "<the exact visible text or appearance of that element>"}}
Answer "found": true ONLY if you can cite concrete evidence that the box
contains the described element. If nothing on this screen corresponds to the
description, you MUST respond {{"found": false, "id": null, "evidence": null}}."""

_REFINE_PROMPT = """You are looking at a zoomed view of one screen region \
(an open dropdown list, form section, or panel) with a red grid overlay. \
Each cell is labeled in its top-left corner (A1, B3, ...).

Target element: {description}

Give the cell range covering the ENTIRE matching element — for a list option
the full height of its text line, for an input field the input line itself
(not its label) — from its top-left cell to its bottom-right cell, for
example "C1:D4". Do not include any part of neighboring rows or elements.
Match by meaning: "FL" matches Florida, but the text must denote EXACTLY the
described item — a different item of the same kind (a different year, state,
or rating) is NOT a match, even if it is the only one visible. Adjacent rows
often look similar (for example "Excellent (740 and above)" directly above
"Very good (680-739)"): read the row texts carefully and double-check the
range covers the matching row, not a neighbor.

Respond with ONLY this JSON, no other text:
{{"found": true, "cells": "<top-left cell>:<bottom-right cell>", "evidence": "<the exact visible text of that row>"}}
Answer "found": true ONLY if you can cite the exact visible text proving the
match. If the target is not in this view, you MUST respond
{{"found": false, "cells": null, "evidence": null}} — NEVER pick a closest or
similar row for a target that is not there."""

AskFn = Callable[[str, Image.Image], Awaitable[Optional[dict]]]

# Descriptions whose click point should anchor to the left edge of a wide
# detection box (the control precedes its label in LTR layouts).
_LEFT_ANCHORED_CONTROL = re.compile(r"\b(radio|check\s?box|toggle|switch)\b", re.IGNORECASE)

# A matched box taller than this (screenshot pixels) covers several text rows
# (e.g. an open dropdown list boxed as one element) and needs a second pass to
# pinpoint the exact row.
MULTI_ROW_BOX_PX = 120

# Lookups for an option inside an open list. The detector frequently does not
# box the list rows at all, so the match degrades to the dropdown control box
# and the real option must be found in the area around it.
_OPTION_DESC = re.compile(r"\boption\b", re.IGNORECASE)


def _evidence_supports(description: str, evidence: str) -> bool:
    """Programmatic guard for hallucinated matches: when the description
    quotes a purely numeric literal (a year, an amount), the cited evidence
    must contain it verbatim — semantic matching can justify "FL" for
    Florida, but never "2026" as evidence for "1999"."""
    m = re.search(r'"([^"]+)"', description)
    if not m:
        return True
    literal = m.group(1).strip()
    if not literal.replace(".", "").replace(",", "").isdigit():
        return True
    return literal in (evidence or "")


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OmniParserLocator:
    def __init__(self) -> None:
        self._model: Optional[YOLO] = None
        self._lock = threading.Lock()
        self._device = _pick_device()
        self.debug_dir = (
            Path(__file__).parent / "debug" if os.getenv("LOCATOR_DEBUG") else None
        )
        # Warm up in the background so the first locate doesn't pay for the
        # weights download and model load.
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self) -> None:
        try:
            self._ensure_model()
        except Exception:
            logger.exception("OmniParser warmup failed")

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    weights = hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE)
                    model = YOLO(weights)
                    logger.info(
                        f"Loaded OmniParser icon detector ({weights}) on {self._device}"
                    )
                    self._model = model
        return self._model

    def detect(self, screenshot: Image.Image) -> List[Tuple[float, float, float, float]]:
        """Detect interactable elements; returns xyxy boxes in screenshot pixels."""
        model = self._ensure_model()
        result = model.predict(
            screenshot.convert("RGB"),
            conf=BOX_CONFIDENCE,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=self._device,
            verbose=False,
        )[0]
        boxes = [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()]
        # Keep the most confident detections when a screen is very busy.
        return boxes[:MAX_ELEMENTS]

    async def locate(
        self,
        description: str,
        screenshot: Image.Image,
        scale_factor: float,
        ask: AskFn,
    ) -> Tuple[Optional[Tuple[float, float]], str]:
        """Find an element; returns ((x, y) in logical points or None, status).

        Status is "matched", "refused" (the model explicitly said the element
        is not on screen — trustworthy, it saw every detected element), or
        "error" (no detections / unusable model response — worth a fallback).

        `ask` is the shared vision-LLM helper from ElementLocator (prompt,
        image) -> parsed JSON dict.
        """
        self._debug_seq = getattr(self, "_debug_seq", 0) + 1
        boxes = await asyncio.to_thread(self.detect, screenshot)
        if not boxes:
            logger.info("OmniParser detected no elements on screen")
            return None, "error"

        som = self._draw_marks(screenshot, boxes)
        self._save_debug(som, "som")

        data = await ask(_SOM_PROMPT.format(description=description), som)
        if data is None:
            return None, "error"
        if not data.get("found") or data.get("id") is None:
            if _OPTION_DESC.search(description):
                # Open-list rows often have no boxes at all, so a refusal for
                # an option lookup is unreliable. Anchor on the dropdown
                # control instead (controls are reliably boxed) and search the
                # region around it, where the open list renders.
                target = await self._locate_option_near_control(
                    description, screenshot, boxes, som, scale_factor, ask
                )
                if target is not None:
                    return target, "matched"
                logger.info(
                    f"OmniParser refused option lookup {description!r}; "
                    f"deferring to grid"
                )
                return None, "error"
            return None, "refused"
        try:
            idx = int(data["id"])
        except (TypeError, ValueError):
            logger.warning(f"OmniParser pick was not a number: {data.get('id')!r}")
            return None, "error"
        if not 0 <= idx < len(boxes):
            logger.warning(f"OmniParser pick out of range: {idx} of {len(boxes)}")
            return None, "error"

        x0, y0, x1, y1 = boxes[idx]
        evidence = str(data.get("evidence") or "")
        height = y1 - y0

        # Open list rows often get NO boxes of their own, so an option lookup
        # degrades to the dropdown control's small box. The list renders
        # adjacent to the control (below, or above near the screen bottom):
        # search that whole region for the option row.
        if _OPTION_DESC.search(description) and height <= MULTI_ROW_BOX_PX and "\n" not in evidence:
            # The pick may be any small box near the list (the control, or a
            # neighboring field), so search generously around it; the refine
            # pass verifies the option is actually present in the region.
            width = x1 - x0
            region = (
                max(0.0, x0 - 1.25 * width),
                max(0.0, y0 - 10 * height),
                min(float(screenshot.width), x1 + 1.25 * width),
                min(float(screenshot.height), y1 + 10 * height),
            )
            refined = await self._refine_in_box(
                description, screenshot, region, scale_factor, ask
            )
            if refined is not None:
                logger.info(
                    f"OmniParser located {description!r}: list region around box "
                    f"{idx} -> logical ({int(refined[0])}, {int(refined[1])})"
                )
                return refined, "matched"
            # The picked box may have been the wrong anchor entirely; retry
            # anchored on the dropdown control, then let the grid have a go.
            target = await self._locate_option_near_control(
                description, screenshot, boxes, som, scale_factor, ask
            )
            if target is not None:
                return target, "matched"
            logger.info(
                f"OmniParser: {description!r} not found near box {idx} or its "
                f"control; deferring to grid"
            )
            return None, "error"

        # An open dropdown often gets detected as ONE box spanning the whole
        # list (the evidence then shows several stacked texts): the box center
        # would hit an arbitrary row (this is how BUICK became BMW). Zoom into
        # the box and pinpoint the exact row instead.
        if height > MULTI_ROW_BOX_PX or "\n" in evidence:
            refined = await self._refine_in_box(
                description, screenshot, (x0, y0, x1, y1), scale_factor, ask
            )
            if refined is not None:
                logger.info(
                    f"OmniParser located {description!r}: box {idx} refined -> "
                    f"logical ({int(refined[0])}, {int(refined[1])})"
                )
                return refined, "matched"

        if not _evidence_supports(description, evidence):
            logger.info(
                f"OmniParser pick rejected: evidence {evidence!r} does not "
                f"contain the quoted literal from {description!r}"
            )
            return None, "refused"

        cx = (x0 + x1) / 2
        # Radio buttons and checkboxes get detected as one box spanning the
        # control AND its text label (often plus trailing whitespace), so the
        # box center can land on empty space. The clickable control sits at
        # the left edge of such a row: click there instead.
        if _LEFT_ANCHORED_CONTROL.search(description) and (x1 - x0) > 2 * height:
            cx = x0 + height / 2
        target = (cx / scale_factor, (y0 + y1) / 2 / scale_factor)
        logger.info(
            f"OmniParser located {description!r}: box {idx} "
            f"(evidence: {evidence!r}) -> logical "
            f"({int(target[0])}, {int(target[1])})"
        )
        return target, "matched"

    async def _locate_option_near_control(
        self,
        description: str,
        screenshot: Image.Image,
        boxes: List[Tuple[float, float, float, float]],
        som: Image.Image,
        scale_factor: float,
        ask: AskFn,
    ) -> Optional[Tuple[float, float]]:
        """Find a list option by locating its dropdown control and searching
        the region around it (where the open list renders, below or above)."""
        m = re.search(
            r"\boption\b\s+(?:in|of|from)\s+(.+?)(?:\s+list)?\s*$",
            description,
            re.IGNORECASE,
        )
        control_desc = m.group(1).strip() if m else "the dropdown that is currently open"

        data = await ask(_SOM_PROMPT.format(description=control_desc), som)
        if not data or not data.get("found") or data.get("id") is None:
            return None
        try:
            idx = int(data["id"])
        except (TypeError, ValueError):
            return None
        if not 0 <= idx < len(boxes):
            return None

        x0, y0, x1, y1 = boxes[idx]
        width, height = x1 - x0, y1 - y0
        region = (
            max(0.0, x0 - 0.25 * width),
            max(0.0, y0 - 10 * height),
            min(float(screenshot.width), x1 + 0.25 * width),
            min(float(screenshot.height), y1 + 10 * height),
        )
        logger.info(
            f"Option lookup anchored on control {control_desc!r} (box {idx}); "
            f"searching region around it"
        )
        return await self._refine_in_box(description, screenshot, region, scale_factor, ask)

    async def _refine_in_box(
        self,
        description: str,
        screenshot: Image.Image,
        box: Tuple[float, float, float, float],
        scale_factor: float,
        ask: AskFn,
    ) -> Optional[Tuple[float, float]]:
        """Pinpoint the target inside a multi-row detection box with a fine grid."""
        x0, y0, x1, y1 = box
        geometry = grid_mod.GridGeometry(
            rows=1,
            cols=1,
            cell_width=x1 - x0,
            cell_height=y1 - y0,
            origin_x=x0,
            origin_y=y0,
            scale_factor=scale_factor,
        )
        fine = grid_mod.crop_with_fine_grid(
            screenshot, geometry, "A1", margin=0.05, rows=10, cols=4
        )
        if fine is None:
            return None
        crop, fine_geometry = fine
        self._save_debug(crop, "refine")

        data = await ask(_REFINE_PROMPT.format(description=description), crop)
        if not data or not data.get("found"):
            return None
        if not _evidence_supports(description, str(data.get("evidence") or "")):
            logger.info(
                f"Refine rejected: evidence {data.get('evidence')!r} does not "
                f"contain the quoted literal from {description!r}"
            )
            return None
        cells = str(data.get("cells") or data.get("cell") or "")
        target = fine_geometry.range_center_logical(cells) if cells else None
        if target is None:
            return None

        # One 10-row pass over a tall list quantizes to ~half a row, which can
        # select the neighboring option. Keep zooming until cells are well
        # below a list row's height (~40-80 screenshot px), so the final range
        # center cannot drift into a neighboring row.
        for pass_num in range(2, 5):
            if fine_geometry.cell_height * fine_geometry.downscale <= 16:
                break
            deeper = grid_mod.crop_with_fine_grid(
                screenshot, fine_geometry, cells, margin=0.5, rows=8, cols=4
            )
            if deeper is None:
                break
            crop, deeper_geometry = deeper
            self._save_debug(crop, f"refine{pass_num}")
            data = await ask(_REFINE_PROMPT.format(description=description), crop)
            if not data or not data.get("found"):
                break
            deeper_cells = str(data.get("cells") or data.get("cell") or "")
            deeper_target = (
                deeper_geometry.range_center_logical(deeper_cells)
                if deeper_cells
                else None
            )
            if deeper_target is None:
                break
            fine_geometry, cells, target = deeper_geometry, deeper_cells, deeper_target

        return target

    def _draw_marks(
        self, screenshot: Image.Image, boxes: List[Tuple[float, float, float, float]]
    ) -> Image.Image:
        som = screenshot.copy()
        draw = ImageDraw.Draw(som, "RGBA")
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except OSError:
            font = ImageFont.load_default()
        for idx, (x0, y0, x1, y1) in enumerate(boxes):
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 220), width=3)
            label = str(idx)
            bbox = draw.textbbox((x0 + 4, y0 + 2), label, font=font)
            draw.rectangle(
                [bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
                fill=(255, 255, 255, 210),
            )
            draw.text((x0 + 4, y0 + 2), label, fill=(255, 0, 0, 255), font=font)
        return som

    def _save_debug(self, image: Image.Image, name: str) -> None:
        if self.debug_dir is None:
            return
        self.debug_dir.mkdir(exist_ok=True)
        seq = getattr(self, "_debug_seq", 0)
        image.save(self.debug_dir / f"locate{seq:03d}_{name}.png", format="PNG")
        image.save(self.debug_dir / f"locate-current.png", format="PNG")

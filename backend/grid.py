"""Screen capture and labeled-grid overlay utilities for visual element location.

Coordinates come in two flavors:
- "screenshot pixels": the raw pyautogui screenshot resolution (2x logical on Retina)
- "logical points": what pyautogui.moveTo / pyautogui.click expect

GridGeometry carries enough information to map a cell label on any grid image
(the full screen, or an upscaled crop of one cell) back to logical screen
coordinates, so nothing about the display needs to be hardcoded.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import pyautogui
from PIL import Image, ImageDraw, ImageFont


# Minimum width (in pixels) a fine-grid crop is rendered at before being sent
# to the vision model; small crops are upscaled to at least this size.
MIN_RENDER_WIDTH = 800


def _load_font(size: int = 24) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


@dataclass
class GridGeometry:
    rows: int
    cols: int
    cell_width: float   # in pixels of the image the grid was drawn on
    cell_height: float
    origin_x: float = 0.0      # top-left of this grid within the full screenshot, in screenshot pixels
    origin_y: float = 0.0
    downscale: float = 1.0     # full-screenshot pixels per drawn-image pixel
    scale_factor: float = 1.0  # screenshot pixels per logical point (2.0 on Retina)

    def parse_label(self, label: str) -> Optional[Tuple[int, int]]:
        """Parse a cell label like 'D5' into (row_idx, col_idx), or None if invalid."""
        label = label.strip().upper()
        if len(label) < 2 or not label[0].isalpha() or not label[1:].isdigit():
            return None
        row = ord(label[0]) - ord("A")
        col = int(label[1:]) - 1
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def parse_range(self, spec: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse a cell or cell range like 'C5' or 'B4:B8' into
        (row0, col0, row1, col1) bounds, or None if invalid."""
        parts = spec.strip().split(":")
        first = self.parse_label(parts[0])
        last = self.parse_label(parts[-1]) if len(parts) > 1 else first
        if first is None or last is None:
            return None
        row0, row1 = sorted((first[0], last[0]))
        col0, col1 = sorted((first[1], last[1]))
        return row0, col0, row1, col1

    def range_center_logical(self, spec: str) -> Optional[Tuple[float, float]]:
        """Center of the cell or cell range in logical screen points, or None.

        Elements often span several cells (a wide input field or button), so
        the click target is the center of the spanned rectangle rather than
        the center of a single cell.
        """
        parsed = self.parse_range(spec)
        if parsed is None:
            return None
        row0, col0, row1, col1 = parsed
        x = self.origin_x + ((col0 + col1 + 1) / 2) * self.cell_width * self.downscale
        y = self.origin_y + ((row0 + row1 + 1) / 2) * self.cell_height * self.downscale
        return x / self.scale_factor, y / self.scale_factor

    def cell_center_logical(self, label: str) -> Optional[Tuple[float, float]]:
        """Center of a single labeled cell in logical screen points, or None."""
        return self.range_center_logical(label)


def screenshot_scale_factor(screenshot: Image.Image) -> float:
    logical_width, _ = pyautogui.size()
    return screenshot.width / logical_width


def capture_screenshot() -> Tuple[Image.Image, float]:
    """Capture the screen. Returns (RGBA image, screenshot-pixels-per-logical-point)."""
    screenshot = pyautogui.screenshot().convert("RGBA")
    return screenshot, screenshot_scale_factor(screenshot)


def draw_grid(
    image: Image.Image,
    rows: int,
    cols: int,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    downscale: float = 1.0,
    scale_factor: float = 1.0,
    line_width: int = 3,
    font_size: int = 24,
) -> GridGeometry:
    """Draw a labeled red grid (A1, A2, ... top-left of each cell) on the image in place."""
    draw = ImageDraw.Draw(image, "RGBA")
    cell_width = image.width / cols
    cell_height = image.height / rows
    font = _load_font(font_size)

    for row in range(rows):
        for col in range(cols):
            x0 = col * cell_width
            y0 = row * cell_height
            draw.rectangle(
                [x0, y0, x0 + cell_width, y0 + cell_height],
                outline=(255, 0, 0, 200),
                width=line_width,
            )
            label = f"{chr(65 + row)}{col + 1}"
            draw.text((x0 + 6, y0 + 4), label, fill=(255, 0, 0, 255), font=font)

    image.save('my_grid-new-1.png', format='PNG')
    return GridGeometry(
        rows=rows,
        cols=cols,
        cell_width=cell_width,
        cell_height=cell_height,
        origin_x=origin_x,
        origin_y=origin_y,
        downscale=downscale,
        scale_factor=scale_factor,
    )


def capture_with_grid(rows: int = 10, cols: int = 10):
    """Capture the screen and overlay a coarse labeled grid.

    Returns (clean screenshot, gridded copy, geometry). The clean screenshot is
    kept so a second, finer pass can crop from an un-annotated image.
    """
    screenshot, scale_factor = capture_screenshot()
    gridded = screenshot.copy()
    geometry = draw_grid(gridded, rows, cols, scale_factor=scale_factor)
    return screenshot, gridded, geometry


def crop_with_fine_grid(
    screenshot: Image.Image,
    geometry: GridGeometry,
    spec: str,
    *,
    margin: float = 1.0,
    rows: int = 8,
    cols: int = 8,
    upscale: int = 2,
):
    """Crop around the labeled coarse cell or cell range (plus margin cells on
    each side), upscale it, and overlay a finer grid for a precise second
    locate pass.

    The full-cell margin means the crop covers the full neighborhood of the
    chosen cells, so the fine pass recovers from off-by-one coarse picks when
    the target sits on a cell boundary.

    The geometry may itself come from an earlier crop pass: the range bounds
    are mapped back to full-screenshot pixels through its origin/downscale, and
    the crop is always taken from the full-resolution screenshot, so passes can
    be chained for progressively finer grids.

    Returns (gridded crop, geometry) or None for an invalid label.
    """
    parsed = geometry.parse_range(spec)
    if parsed is None:
        return None
    row0, col0, row1, col1 = parsed

    # Cell size of the current grid, in full-screenshot pixels.
    cell_w = geometry.cell_width * geometry.downscale
    cell_h = geometry.cell_height * geometry.downscale

    x0 = max(0.0, geometry.origin_x + (col0 - margin) * cell_w)
    y0 = max(0.0, geometry.origin_y + (row0 - margin) * cell_h)
    x1 = min(float(screenshot.width), geometry.origin_x + (col1 + 1 + margin) * cell_w)
    y1 = min(float(screenshot.height), geometry.origin_y + (row1 + 1 + margin) * cell_h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None

    crop = screenshot.crop((int(x0), int(y0), int(x1), int(y1)))

    # Upscale small crops aggressively (deep zoom passes can be only a couple
    # hundred pixels wide): without this the grid labels dwarf the cells and
    # bury the target under red text, leaving the model to guess.
    scale = max(upscale, -(-MIN_RENDER_WIDTH // max(1, crop.width)))  # ceil div
    scale = min(scale, 10)
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)

    cell_min = min(crop.width / cols, crop.height / rows)
    font_size = int(min(28, max(12, cell_min * 0.22)))

    fine_geometry = draw_grid(
        crop,
        rows,
        cols,
        origin_x=float(int(x0)),
        origin_y=float(int(y0)),
        downscale=1.0 / scale,
        scale_factor=geometry.scale_factor,
        line_width=2,
        font_size=font_size,
    )
    return crop, fine_geometry

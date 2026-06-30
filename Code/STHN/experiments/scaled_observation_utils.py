"""
Utilities for simulating scaled partial thermal observations.

The intended simulation is:
  - a normal thermal observation fills the model input canvas;
  - if the UAV changes altitude, each observation covers a smaller effective
    footprint on that standard canvas;
  - several smaller observations are pasted into one canvas;
  - regions not covered by any observation become no-data areas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MosaicMetadata:
    scale: float
    coverage_ratio: float
    missing_ratio: float
    placements: list[tuple[int, int, int, int]]


def _fill_canvas_like(image: torch.Tensor, fill_mode: str, fill_value: float) -> torch.Tensor:
    if fill_mode == "zero":
        return torch.zeros_like(image)
    if fill_mode == "half":
        return torch.full_like(image, 0.5)
    if fill_mode == "mean":
        mean = image.mean(dim=(-2, -1), keepdim=True)
        return mean.expand_as(image).clone()
    if fill_mode == "value":
        return torch.full_like(image, float(fill_value))
    raise ValueError(f"Unknown fill_mode: {fill_mode}")


def _path_positions(
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    num_tiles: int,
    path: str,
    jitter: int,
    generator: torch.Generator | None,
) -> list[tuple[int, int]]:
    max_y = max(0, height - tile_h)
    max_x = max(0, width - tile_w)

    if num_tiles <= 1:
        positions = [(max_y // 2, max_x // 2)]
    elif path == "diagonal":
        positions = [
            (
                round(max_y * i / (num_tiles - 1)),
                round(max_x * i / (num_tiles - 1)),
            )
            for i in range(num_tiles)
        ]
    elif path == "horizontal":
        positions = [
            (
                max_y // 2,
                round(max_x * i / (num_tiles - 1)),
            )
            for i in range(num_tiles)
        ]
    elif path == "vertical":
        positions = [
            (
                round(max_y * i / (num_tiles - 1)),
                max_x // 2,
            )
            for i in range(num_tiles)
        ]
    elif path == "corners":
        base = [(0, 0), (0, max_x), (max_y, 0), (max_y, max_x), (max_y // 2, max_x // 2)]
        positions = base[:num_tiles]
        if len(positions) < num_tiles:
            positions.extend([base[-1]] * (num_tiles - len(positions)))
    elif path == "random":
        positions = []
        for _ in range(num_tiles):
            y = int(torch.randint(0, max_y + 1, (1,), generator=generator).item()) if max_y else 0
            x = int(torch.randint(0, max_x + 1, (1,), generator=generator).item()) if max_x else 0
            positions.append((y, x))
    else:
        raise ValueError(f"Unknown path: {path}")

    if jitter > 0 and path != "random":
        jittered = []
        for y, x in positions:
            dy = int(torch.randint(-jitter, jitter + 1, (1,), generator=generator).item())
            dx = int(torch.randint(-jitter, jitter + 1, (1,), generator=generator).item())
            jittered.append((min(max(y + dy, 0), max_y), min(max(x + dx, 0), max_x)))
        positions = jittered

    return positions


def make_scaled_observation_mosaic(
    image: torch.Tensor,
    scale: float,
    num_tiles: int = 2,
    path: str = "diagonal",
    fill_mode: str = "zero",
    fill_value: float = 0.0,
    blend: str = "average",
    jitter: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, MosaicMetadata]:
    """Create a partial-observation mosaic from a batch of images.

    Args:
        image: Tensor shaped [B, C, H, W], values usually in [0, 1].
        scale: Linear scale of each observation inside the standard canvas.
            Example: 0.8 means each tile is 80% of H/W.
        num_tiles: Number of smaller observations to paste.
        path: Placement pattern: diagonal, horizontal, vertical, corners, random.
        fill_mode: Value used for no-data regions: zero, half, mean, value.
        blend: "average" averages overlap; "overwrite" uses later tiles.
        jitter: Pixel jitter added to deterministic paths.

    Returns:
        mosaic tensor and metadata.
    """
    if image.ndim != 4:
        raise ValueError("Expected image tensor [B, C, H, W]")
    if not 0.05 <= scale <= 1.0:
        raise ValueError("Expected scale in [0.05, 1.0]")
    if num_tiles < 1:
        raise ValueError("num_tiles must be >= 1")

    batch, channels, height, width = image.shape
    tile_h = max(1, min(height, int(round(height * scale))))
    tile_w = max(1, min(width, int(round(width * scale))))
    scaled = F.interpolate(image, size=(tile_h, tile_w), mode="bilinear", align_corners=True)

    canvas = _fill_canvas_like(image, fill_mode, fill_value)
    counts = torch.zeros((batch, 1, height, width), dtype=image.dtype, device=image.device)
    sums = torch.zeros_like(image)
    placements_xyxy: list[tuple[int, int, int, int]] = []

    positions = _path_positions(height, width, tile_h, tile_w, num_tiles, path, jitter, generator)
    for y0, x0 in positions:
        y1 = y0 + tile_h
        x1 = x0 + tile_w
        placements_xyxy.append((x0, y0, x1, y1))
        if blend == "average":
            sums[:, :, y0:y1, x0:x1] += scaled
            counts[:, :, y0:y1, x0:x1] += 1
        elif blend == "overwrite":
            canvas[:, :, y0:y1, x0:x1] = scaled
            counts[:, :, y0:y1, x0:x1] = 1
        else:
            raise ValueError(f"Unknown blend: {blend}")

    if blend == "average":
        covered = counts > 0
        canvas = torch.where(covered.expand_as(canvas), sums / counts.clamp_min(1), canvas)

    coverage_ratio = float((counts > 0).float().mean().item())
    metadata = MosaicMetadata(
        scale=float(scale),
        coverage_ratio=coverage_ratio,
        missing_ratio=1.0 - coverage_ratio,
        placements=placements_xyxy,
    )
    return canvas, metadata


def scale_from_closer_percent(percent_closer: float) -> float:
    """Convert a simple '20% closer means 20% smaller footprint' convention."""
    return max(0.05, min(1.0, 1.0 - percent_closer / 100.0))


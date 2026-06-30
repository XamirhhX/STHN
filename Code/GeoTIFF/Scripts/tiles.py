from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from typing import Iterable

import mercantile
from PIL import Image, UnidentifiedImageError
import requests
from tqdm import tqdm

from .defaults import DEFAULT_TILE_URL_TEMPLATE, DEFAULT_USER_AGENT
from .geo import (
    Bounds3857,
    TILE_SIZE_PX,
    bounds_3857_to_4326,
    global_pixel_to_web_mercator,
    web_mercator_to_global_pixel,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileGrid:
    tiles: list[mercantile.Tile]
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    zoom: int

    @property
    def width_tiles(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height_tiles(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def width_px(self) -> int:
        return self.width_tiles * TILE_SIZE_PX

    @property
    def height_px(self) -> int:
        return self.height_tiles * TILE_SIZE_PX


@dataclass(frozen=True)
class CropInfo:
    left_px: int
    top_px: int
    right_px: int
    bottom_px: int
    bounds_3857: Bounds3857
    width_px: int
    height_px: int


def required_tiles(bounds_3857: Bounds3857, zoom: int) -> TileGrid:
    bounds_4326 = bounds_3857_to_4326(bounds_3857)
    tiles = list(
        mercantile.tiles(
            bounds_4326.west,
            bounds_4326.south,
            bounds_4326.east,
            bounds_4326.north,
            zooms=[zoom],
        )
    )
    if not tiles:
        raise ValueError("No map tiles found for requested bounds.")
    x_values = [tile.x for tile in tiles]
    y_values = [tile.y for tile in tiles]
    return TileGrid(
        tiles=tiles,
        min_x=min(x_values),
        max_x=max(x_values),
        min_y=min(y_values),
        max_y=max(y_values),
        zoom=zoom,
    )


def tile_url(template: str, tile: mercantile.Tile) -> str:
    return template.format(
        z=tile.z,
        x=tile.x,
        y=tile.y,
        quadkey=mercantile.quadkey(tile),
    )


def tile_cache_path(cache_dir: Path, tile: mercantile.Tile) -> Path:
    return cache_dir / str(tile.z) / str(tile.x) / f"{tile.y}.png"


def load_cached_tile(cache_dir: Path, tile: mercantile.Tile) -> Image.Image | None:
    path = tile_cache_path(cache_dir, tile)
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            loaded = image.convert("RGB")
            loaded.load()
            return loaded
    except (OSError, UnidentifiedImageError) as exc:
        LOGGER.warning("Ignoring unreadable cached tile %s: %s", path, exc)
        return None


def save_cached_tile(cache_dir: Path, tile: mercantile.Tile, image: Image.Image) -> None:
    path = tile_cache_path(cache_dir, tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def download_tile(
    session: requests.Session,
    tile: mercantile.Tile,
    url_template: str,
    cache_dir: Path,
    retries: int,
    timeout_seconds: float,
    use_cache: bool,
) -> Image.Image | None:
    if use_cache:
        cached = load_cached_tile(cache_dir, tile)
        if cached is not None:
            return cached

    url = tile_url(url_template, tile)
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as image:
                loaded = image.convert("RGB")
                if loaded.size != (TILE_SIZE_PX, TILE_SIZE_PX):
                    LOGGER.warning("Resizing non-standard tile %s from %s.", tile, loaded.size)
                    loaded = loaded.resize((TILE_SIZE_PX, TILE_SIZE_PX))
                loaded.load()
            if use_cache:
                save_cached_tile(cache_dir, tile, loaded)
            return loaded
        except (requests.RequestException, OSError, UnidentifiedImageError) as exc:
            if attempt >= retries:
                LOGGER.warning("Tile %s failed after %d attempts: %s", tile, retries + 1, exc)
                return None
            sleep_seconds = min(2.0**attempt, 8.0)
            LOGGER.debug("Retrying tile %s in %.1fs after error: %s", tile, sleep_seconds, exc)
            time.sleep(sleep_seconds)
    return None


def download_tiles(
    grid: TileGrid,
    url_template: str,
    cache_dir: Path,
    retries: int = 3,
    timeout_seconds: float = 20.0,
    user_agent: str = DEFAULT_USER_AGENT,
    use_cache: bool = True,
) -> dict[mercantile.Tile, Image.Image | None]:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    images: dict[mercantile.Tile, Image.Image | None] = {}
    progress = tqdm(grid.tiles, desc="Downloading tiles", unit="tile")
    for tile in progress:
        images[tile] = download_tile(
            session=session,
            tile=tile,
            url_template=url_template,
            cache_dir=cache_dir,
            retries=retries,
            timeout_seconds=timeout_seconds,
            use_cache=use_cache,
        )
    return images


def blank_tile(fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    return Image.new("RGB", (TILE_SIZE_PX, TILE_SIZE_PX), color=fill)


def stitch_tiles(grid: TileGrid, images: dict[mercantile.Tile, Image.Image | None]) -> Image.Image:
    canvas = Image.new("RGB", (grid.width_px, grid.height_px), color=(0, 0, 0))
    missing = 0
    for tile in grid.tiles:
        image = images.get(tile)
        if image is None:
            image = blank_tile()
            missing += 1
        x_offset = (tile.x - grid.min_x) * TILE_SIZE_PX
        y_offset = (tile.y - grid.min_y) * TILE_SIZE_PX
        canvas.paste(image, (x_offset, y_offset))
    if missing:
        LOGGER.warning("Stitched image contains %d missing tile(s) filled with black.", missing)
    return canvas


def crop_info_for_bounds(grid: TileGrid, bounds_3857: Bounds3857) -> CropInfo:
    left_world_px, bottom_world_py = web_mercator_to_global_pixel(
        bounds_3857.left,
        bounds_3857.bottom,
        grid.zoom,
    )
    right_world_px, top_world_py = web_mercator_to_global_pixel(
        bounds_3857.right,
        bounds_3857.top,
        grid.zoom,
    )

    canvas_left_world_px = grid.min_x * TILE_SIZE_PX
    canvas_top_world_py = grid.min_y * TILE_SIZE_PX

    left_px = max(0, floor(left_world_px - canvas_left_world_px))
    top_px = max(0, floor(top_world_py - canvas_top_world_py))
    right_px = min(grid.width_px, ceil(right_world_px - canvas_left_world_px))
    bottom_px = min(grid.height_px, ceil(bottom_world_py - canvas_top_world_py))

    if right_px <= left_px or bottom_px <= top_px:
        raise ValueError("Computed crop is empty. Check center, area, and zoom.")

    snapped_left_global_px = canvas_left_world_px + left_px
    snapped_top_global_py = canvas_top_world_py + top_px
    snapped_right_global_px = canvas_left_world_px + right_px
    snapped_bottom_global_py = canvas_top_world_py + bottom_px

    left_m, top_m = global_pixel_to_web_mercator(
        snapped_left_global_px,
        snapped_top_global_py,
        grid.zoom,
    )
    right_m, bottom_m = global_pixel_to_web_mercator(
        snapped_right_global_px,
        snapped_bottom_global_py,
        grid.zoom,
    )
    snapped_bounds = Bounds3857(left=left_m, bottom=bottom_m, right=right_m, top=top_m)

    return CropInfo(
        left_px=left_px,
        top_px=top_px,
        right_px=right_px,
        bottom_px=bottom_px,
        bounds_3857=snapped_bounds,
        width_px=right_px - left_px,
        height_px=bottom_px - top_px,
    )


def crop_stitched_image(stitched: Image.Image, crop_info: CropInfo) -> Image.Image:
    return stitched.crop(
        (
            crop_info.left_px,
            crop_info.top_px,
            crop_info.right_px,
            crop_info.bottom_px,
        )
    )


def count_missing_tiles(images: Iterable[Image.Image | None]) -> int:
    return sum(1 for image in images if image is None)

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from .defaults import DEFAULT_TILE_URL_TEMPLATE, DEFAULT_USER_AGENT
from .geo import (
    Bounds3857,
    area_bounds_around_center,
    bounds_3857_to_4326,
)
from .tiles import (
    count_missing_tiles,
    crop_info_for_bounds,
    crop_stitched_image,
    download_tiles,
    required_tiles,
    stitch_tiles,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoTiffBuildResult:
    geotiff_path: Path
    preview_path: Path
    metadata_path: Path
    output_folder: Path
    width_px: int
    height_px: int
    bounds_3857: Bounds3857
    resolution_m_per_pixel: float


def write_geotiff(
    image: Image.Image,
    output_path: Path,
    bounds_3857: Bounds3857,
) -> float:
    rgb = image.convert("RGB")
    data = np.asarray(rgb, dtype=np.uint8)
    height, width, channels = data.shape
    if channels != 3:
        raise ValueError("Expected RGB image data.")

    x_resolution = bounds_3857.width / float(width)
    y_resolution = bounds_3857.height / float(height)
    resolution = (abs(x_resolution) + abs(y_resolution)) / 2.0
    transform = from_origin(bounds_3857.left, bounds_3857.top, x_resolution, y_resolution)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
        compress="deflate",
        predictor=2,
    ) as dataset:
        for band_index in range(3):
            dataset.write(data[:, :, band_index], band_index + 1)
        dataset.update_tags(
            AREA_OR_POINT="Area",
            source="geotiff-scripts",
        )
    return resolution


def metadata_payload(
    center_lat: float,
    center_lon: float,
    area_size_meters: float,
    zoom: int,
    requested_bounds_3857: Bounds3857,
    raster_bounds_3857: Bounds3857,
    width_px: int,
    height_px: int,
    tile_count: int,
    missing_tile_count: int,
    resolution_m_per_pixel: float,
    tile_url_template: str,
    output_size_px: int | None,
) -> dict[str, Any]:
    raster_bounds_4326 = bounds_3857_to_4326(raster_bounds_3857)
    requested_bounds_4326 = bounds_3857_to_4326(requested_bounds_3857)
    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "area_size_meters": area_size_meters,
        "zoom": zoom,
        "crs": "EPSG:3857",
        "bounds": {
            "epsg3857": raster_bounds_3857.to_dict(),
            "epsg4326": raster_bounds_4326.to_dict(),
        },
        "requested_bounds": {
            "epsg3857": requested_bounds_3857.to_dict(),
            "epsg4326": requested_bounds_4326.to_dict(),
        },
        "width_px": width_px,
        "height_px": height_px,
        "resolution_m_per_pixel": resolution_m_per_pixel,
        "output_size_px": output_size_px,
        "tile_count": tile_count,
        "missing_tile_count": missing_tile_count,
        "tile_url_template": tile_url_template,
        "pixel_grid_note": (
            "Raster bounds match the requested STHN chip extent when output_size_px is set; "
            "otherwise bounds are snapped to the EPSG:3857 Web Mercator pixel grid at the requested zoom."
        ),
    }


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_geotiff(
    center_lat: float,
    center_lon: float,
    output_folder: Path,
    area_size_meters: float = 500.0,
    zoom: int = 18,
    tile_url_template: str = DEFAULT_TILE_URL_TEMPLATE,
    retries: int = 3,
    timeout_seconds: float = 20.0,
    user_agent: str = DEFAULT_USER_AGENT,
    max_tiles: int = 256,
    use_cache: bool = True,
    output_size_px: int | None = None,
) -> GeoTiffBuildResult:
    output_folder = output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    if output_size_px is not None and output_size_px <= 0:
        raise ValueError("output_size_px must be positive.")
    requested_bounds = area_bounds_around_center(center_lat, center_lon, area_size_meters)
    grid = required_tiles(requested_bounds, zoom)
    if len(grid.tiles) > max_tiles:
        msg = (
            f"Requested area requires {len(grid.tiles)} tiles, above --max-tiles {max_tiles}. "
            "Use a smaller area, lower zoom, or raise the limit intentionally."
        )
        raise ValueError(msg)

    LOGGER.info(
        "Downloading %d tile(s) at zoom %d for %.1fm area.",
        len(grid.tiles),
        zoom,
        area_size_meters,
    )
    cache_dir = output_folder / "Tiles"
    downloaded = download_tiles(
        grid=grid,
        url_template=tile_url_template,
        cache_dir=cache_dir,
        retries=retries,
        timeout_seconds=timeout_seconds,
        user_agent=user_agent,
        use_cache=use_cache,
    )
    stitched = stitch_tiles(grid, downloaded)
    crop_info = crop_info_for_bounds(grid, requested_bounds)
    cropped = crop_stitched_image(stitched, crop_info)
    raster_bounds = crop_info.bounds_3857
    if output_size_px is not None:
        raster_bounds = requested_bounds
        if cropped.size != (output_size_px, output_size_px):
            cropped = cropped.resize((output_size_px, output_size_px), Image.Resampling.LANCZOS)

    preview_path = output_folder / "Preview" / "satellite_preview.png"
    geotiff_path = output_folder / "GeoTIFF" / "satellite.tif"
    metadata_path = output_folder / "Metadata" / "geotiff_metadata.json"

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(preview_path)
    resolution = write_geotiff(
        image=cropped,
        output_path=geotiff_path,
        bounds_3857=raster_bounds,
    )
    payload = metadata_payload(
        center_lat=center_lat,
        center_lon=center_lon,
        area_size_meters=area_size_meters,
        zoom=zoom,
        requested_bounds_3857=requested_bounds,
        raster_bounds_3857=raster_bounds,
        width_px=cropped.width,
        height_px=cropped.height,
        tile_count=len(grid.tiles),
        missing_tile_count=count_missing_tiles(downloaded.values()),
        resolution_m_per_pixel=resolution,
        tile_url_template=tile_url_template,
        output_size_px=output_size_px,
    )
    write_metadata(metadata_path, payload)
    LOGGER.info("Wrote GeoTIFF: %s", geotiff_path)
    LOGGER.info("Wrote preview: %s", preview_path)
    LOGGER.info("Wrote metadata: %s", metadata_path)
    return GeoTiffBuildResult(
        geotiff_path=geotiff_path,
        preview_path=preview_path,
        metadata_path=metadata_path,
        output_folder=output_folder,
        width_px=cropped.width,
        height_px=cropped.height,
        bounds_3857=raster_bounds,
        resolution_m_per_pixel=resolution,
    )

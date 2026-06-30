from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window, bounds as window_bounds

from .defaults import DEFAULT_SPLIT_RATIOS, SPLIT_NAMES
from .geo import bounds_3857_to_4326, web_mercator_to_lonlat, Bounds3857

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    split: str
    image_path: Path
    metadata_path: Path
    row: int
    col: int
    patch_size: int
    center_lat: float
    center_lon: float
    bbox_epsg3857: Bounds3857
    resolution_m_per_pixel: float

    def to_manifest_dict(self, dataset_folder: Path) -> dict[str, Any]:
        bbox_4326 = bounds_3857_to_4326(self.bbox_epsg3857)
        return {
            "id": self.patch_id,
            "split": self.split,
            "image": self.image_path.relative_to(dataset_folder).as_posix(),
            "metadata": self.metadata_path.relative_to(dataset_folder).as_posix(),
            "row": self.row,
            "col": self.col,
            "patch_size": self.patch_size,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "bbox": {
                "epsg3857": self.bbox_epsg3857.to_dict(),
                "epsg4326": bbox_4326.to_dict(),
            },
            "resolution_m_per_pixel": self.resolution_m_per_pixel,
        }


def parse_split_ratios(raw: str | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(raw, str):
        parts = [float(part.strip()) for part in raw.split(",")]
    else:
        parts = [float(part) for part in raw]
    if len(parts) != 3:
        raise ValueError("Split ratios must contain train,val,test values.")
    total = sum(parts)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return tuple(part / total for part in parts)  # type: ignore[return-value]


def start_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if length < patch_size:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


def split_for_index(index: int, total: int, ratios: tuple[float, float, float]) -> str:
    train_cut = round(total * ratios[0])
    val_cut = train_cut + round(total * ratios[1])
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def raster_window_to_rgb(dataset: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    band_count = min(dataset.count, 3)
    data = dataset.read(indexes=list(range(1, band_count + 1)), window=window)
    if band_count == 1:
        data = np.repeat(data, 3, axis=0)
    if band_count == 2:
        third = np.zeros_like(data[:1])
        data = np.concatenate([data, third], axis=0)
    return np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)


def patch_metadata(
    patch_id: str,
    split: str,
    source_geotiff: Path,
    image_path: Path,
    row: int,
    col: int,
    patch_size: int,
    center_lat: float,
    center_lon: float,
    bbox_epsg3857: Bounds3857,
    resolution_m_per_pixel: float,
    dataset_folder: Path,
) -> dict[str, Any]:
    bbox_4326 = bounds_3857_to_4326(bbox_epsg3857)
    return {
        "id": patch_id,
        "split": split,
        "source_geotiff": str(source_geotiff),
        "image": image_path.relative_to(dataset_folder).as_posix(),
        "row": row,
        "col": col,
        "patch_size": patch_size,
        "center_gps": {
            "lat": center_lat,
            "lon": center_lon,
        },
        "bbox": {
            "epsg3857": bbox_epsg3857.to_dict(),
            "epsg4326": bbox_4326.to_dict(),
        },
        "resolution_m_per_pixel": resolution_m_per_pixel,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def generate_patches(
    geotiff_path: Path,
    output_folder: Path,
    patch_size: int = 512,
    stride: int | None = None,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 1337,
) -> list[PatchRecord]:
    if patch_size not in {512, 1024}:
        LOGGER.warning("Patch size %d is allowed, but 512 or 1024 are recommended.", patch_size)
    geotiff_path = geotiff_path.resolve()
    output_folder = output_folder.resolve()
    image_dir = output_folder / "images"
    metadata_dir = output_folder / "metadata"
    for split_name in SPLIT_NAMES:
        (output_folder / split_name).mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    effective_stride = stride or patch_size
    records: list[PatchRecord] = []
    with rasterio.open(geotiff_path) as dataset:
        if dataset.crs is None or dataset.crs.to_string() != "EPSG:3857":
            LOGGER.warning("Expected EPSG:3857 GeoTIFF, found %s.", dataset.crs)
        rows = start_positions(dataset.height, patch_size, effective_stride)
        cols = start_positions(dataset.width, patch_size, effective_stride)
        windows = [(row, col) for row in rows for col in cols]
        if not windows:
            LOGGER.warning(
                "GeoTIFF %s is %dx%d px, smaller than patch size %d. No patches generated.",
                geotiff_path,
                dataset.width,
                dataset.height,
                patch_size,
            )
            return []

        shuffled_indexes = list(range(len(windows)))
        random.Random(seed).shuffle(shuffled_indexes)
        split_by_original_index = {
            original_index: split_for_index(shuffled_index, len(windows), split_ratios)
            for shuffled_index, original_index in enumerate(shuffled_indexes)
        }

        x_resolution = abs(float(dataset.transform.a))
        y_resolution = abs(float(dataset.transform.e))
        resolution = (x_resolution + y_resolution) / 2.0

        for original_index, (row, col) in enumerate(windows):
            patch_id = f"patch_{original_index:06d}"
            split = split_by_original_index[original_index]
            window = Window(col_off=col, row_off=row, width=patch_size, height=patch_size)
            rgb = raster_window_to_rgb(dataset, window)

            image_path = image_dir / f"{patch_id}.png"
            metadata_path = metadata_dir / f"{patch_id}.json"
            Image.fromarray(rgb, mode="RGB").save(image_path)

            left, bottom, right, top = window_bounds(window, dataset.transform)
            center_x = (left + right) / 2.0
            center_y = (bottom + top) / 2.0
            center_lon, center_lat = web_mercator_to_lonlat(center_x, center_y)
            bbox = Bounds3857(left=left, bottom=bottom, right=right, top=top)

            metadata = patch_metadata(
                patch_id=patch_id,
                split=split,
                source_geotiff=geotiff_path,
                image_path=image_path,
                row=row,
                col=col,
                patch_size=patch_size,
                center_lat=center_lat,
                center_lon=center_lon,
                bbox_epsg3857=bbox,
                resolution_m_per_pixel=resolution,
                dataset_folder=output_folder,
            )
            write_json(metadata_path, metadata)
            records.append(
                PatchRecord(
                    patch_id=patch_id,
                    split=split,
                    image_path=image_path,
                    metadata_path=metadata_path,
                    row=row,
                    col=col,
                    patch_size=patch_size,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    bbox_epsg3857=bbox,
                    resolution_m_per_pixel=resolution,
                )
            )

    all_manifest_rows = [record.to_manifest_dict(output_folder) for record in records]
    write_jsonl(metadata_dir / "patches.jsonl", all_manifest_rows)
    for split_name in SPLIT_NAMES:
        split_rows = [row for row in all_manifest_rows if row["split"] == split_name]
        write_jsonl(output_folder / split_name / "manifest.jsonl", split_rows)

    LOGGER.info("Generated %d patch(es) in %s.", len(records), output_folder)
    return records

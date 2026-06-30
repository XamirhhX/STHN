from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from .defaults import SPLIT_NAMES

LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_patch_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def export_split_to_h5(
    dataset_folder: Path,
    split_name: str,
    output_path: Path,
    compression: str | None = "gzip",
) -> Path:
    manifest_path = dataset_folder / split_name / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        images = np.stack([load_patch_image(dataset_folder / row["image"]) for row in rows], axis=0)
        gps = np.asarray([[row["center_lat"], row["center_lon"]] for row in rows], dtype=np.float64)
        bbox_epsg3857 = np.asarray(
            [
                [
                    row["bbox"]["epsg3857"]["left"],
                    row["bbox"]["epsg3857"]["bottom"],
                    row["bbox"]["epsg3857"]["right"],
                    row["bbox"]["epsg3857"]["top"],
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        bbox_epsg4326 = np.asarray(
            [
                [
                    row["bbox"]["epsg4326"]["west"],
                    row["bbox"]["epsg4326"]["south"],
                    row["bbox"]["epsg4326"]["east"],
                    row["bbox"]["epsg4326"]["north"],
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        ids = np.asarray([row["id"].encode("utf-8") for row in rows])
        patch_size = int(rows[0]["patch_size"])
    else:
        images = np.empty((0, 0, 0, 3), dtype=np.uint8)
        gps = np.empty((0, 2), dtype=np.float64)
        bbox_epsg3857 = np.empty((0, 4), dtype=np.float64)
        bbox_epsg4326 = np.empty((0, 4), dtype=np.float64)
        ids = np.empty((0,), dtype="S1")
        patch_size = 0

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("images", data=images, compression=compression if len(rows) else None)
        handle.create_dataset("gps", data=gps)
        handle.create_dataset("bbox_epsg3857", data=bbox_epsg3857)
        handle.create_dataset("bbox_epsg4326", data=bbox_epsg4326)
        handle.create_dataset("ids", data=ids)
        handle.attrs["split"] = split_name
        handle.attrs["crs"] = "EPSG:3857"
        handle.attrs["count"] = len(rows)
        handle.attrs["patch_size"] = patch_size
    LOGGER.info("Wrote %s with %d patch(es).", output_path, len(rows))
    return output_path


def export_h5(
    dataset_folder: Path,
    output_folder: Path | None = None,
    compression: str | None = "gzip",
) -> list[Path]:
    dataset_folder = dataset_folder.resolve()
    output_folder = (output_folder or dataset_folder).resolve()
    output_paths: list[Path] = []
    for split_name in SPLIT_NAMES:
        output_path = output_folder / f"{split_name}.h5"
        output_paths.append(
            export_split_to_h5(
                dataset_folder=dataset_folder,
                split_name=split_name,
                output_path=output_path,
                compression=compression,
            )
        )
    return output_paths

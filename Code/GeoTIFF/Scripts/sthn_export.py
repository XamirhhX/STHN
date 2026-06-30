from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from PIL import Image, ImageOps

from .defaults import DEFAULT_SPLIT_RATIOS, SPLIT_NAMES

LOGGER = logging.getLogger(__name__)

DEFAULT_STHN_DATASET_NAME = "satellite_0_thermalmapping_135_train"
STHN_MAP_RELATIVE_PATH = Path("maps") / "satellite" / "20201117_BingSatellite.png"


@dataclass(frozen=True)
class STHNExportResult:
    output_folder: Path
    dataset_root: Path
    dataset_name: str
    map_path: Path
    metadata_path: Path
    readme_path: Path
    sample_count: int


def read_geotiff_rgb(path: Path) -> Image.Image:
    with rasterio.open(path) as dataset:
        band_count = min(dataset.count, 3)
        data = dataset.read(indexes=list(range(1, band_count + 1)))
        if band_count == 1:
            data = np.repeat(data, 3, axis=0)
        elif band_count == 2:
            data = np.concatenate([data, np.zeros_like(data[:1])], axis=0)
    rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def split_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    if total <= 0:
        return {split: 0 for split in SPLIT_NAMES}
    if total == 1:
        return {"train": 0, "val": 0, "test": 1}
    if total == 2:
        return {"train": 1, "val": 0, "test": 1}

    train_count = max(1, int(round(total * ratios[0])))
    val_count = max(1, int(round(total * ratios[1])))
    test_count = total - train_count - val_count
    while test_count < 1 and train_count > 1:
        train_count -= 1
        test_count += 1
    while test_count < 1 and val_count > 1:
        val_count -= 1
        test_count += 1
    return {"train": train_count, "val": val_count, "test": test_count}


def assign_splits(total: int, ratios: tuple[float, float, float], seed: int) -> dict[int, str]:
    counts = split_counts(total, ratios)
    labels = ["train"] * counts["train"] + ["val"] * counts["val"] + ["test"] * counts["test"]
    indexes = list(range(total))
    random.Random(seed).shuffle(indexes)
    return {index: labels[rank] for rank, index in enumerate(indexes)}


def sample_centers(width: int, height: int, database_size: int, stride: int) -> list[tuple[int, int]]:
    half = database_size // 2
    if width < database_size or height < database_size:
        raise ValueError(
            f"GeoTIFF image is {width}x{height}px, smaller than STHN database_size {database_size}px."
        )

    def axis_centers(length: int) -> list[int]:
        end = length - half
        centers = list(range(half, end + 1, stride))
        if centers[-1] != end:
            centers.append(end)
        return centers

    xs = axis_centers(width)
    ys = axis_centers(height)
    return [(x, y) for y in ys for x in xs]


def crop_center(image: Image.Image, x: int, y: int, size: int) -> Image.Image:
    half = size // 2
    return image.crop((x - half, y - half, x + half, y + half))


def make_synthetic_query(chip: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(chip)
    gray = ImageOps.autocontrast(gray)
    return Image.merge("RGB", (gray, gray, gray))


def h5_compression_kwargs(compression: str | None) -> dict[str, str]:
    return {"compression": compression} if compression else {}


def write_sthn_h5(
    path: Path,
    image_names: list[str],
    images: list[np.ndarray],
    image_size: int,
    compression: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if images:
        image_data = np.stack(images, axis=0).astype(np.uint8)
    else:
        image_data = np.empty((0, image_size, image_size, 3), dtype=np.uint8)

    sizes = np.full((len(image_names), 2), image_size, dtype=np.int32)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as handle:
        image_kwargs = h5_compression_kwargs(compression)
        if len(image_data):
            handle.create_dataset(
                "image_data",
                data=image_data,
                chunks=(1, image_size, image_size, 3),
                maxshape=(None, image_size, image_size, 3),
                **image_kwargs,
            )
        else:
            handle.create_dataset("image_data", data=image_data)
        handle.create_dataset("image_size", data=sizes, maxshape=(None, 2))
        handle.create_dataset("image_name", data=image_names, dtype=string_dtype)


def write_readme(
    path: Path,
    dataset_root: Path,
    dataset_name: str,
    database_size: int,
    query_size: int,
) -> None:
    text = f"""# STHN Model Input

This folder is formatted for the STHN loaders.

Use it with the STHN repo like this:

```powershell
python experiments/missing_data_eval.py `
  --datasets_folder "{dataset_root}" `
  --dataset_name {dataset_name} `
  --split test `
  --database_size {database_size} `
  --crop_width {query_size} `
  --batch_size 1 `
  --num_workers 0
```

Important: the query H5 files contain synthetic grayscale query chips derived from the satellite GeoTIFF. They are format-compatible with the model pipeline, but they are not real thermal imagery and should not be treated as real satellite-thermal evaluation data.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_sthn_dataset(
    geotiff_path: Path,
    output_folder: Path,
    dataset_name: str = DEFAULT_STHN_DATASET_NAME,
    database_size: int = 1536,
    query_size: int = 512,
    stride: int | None = None,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 1337,
    compression: str | None = "gzip",
) -> STHNExportResult:
    geotiff_path = geotiff_path.resolve()
    output_folder = output_folder.resolve()
    if database_size <= 0 or database_size % 2:
        raise ValueError("database_size must be a positive even integer.")
    if query_size <= 0 or query_size % 2:
        raise ValueError("query_size must be a positive even integer.")

    effective_stride = stride or query_size
    if effective_stride <= 0:
        raise ValueError("stride must be positive.")

    source_image = read_geotiff_rgb(geotiff_path)
    width, height = source_image.size
    centers = sample_centers(width, height, database_size, effective_stride)
    split_by_index = assign_splits(len(centers), split_ratios, seed)

    dataset_root = output_folder / "sthn_dataset"
    dataset_folder = dataset_root / dataset_name
    map_path = dataset_root / STHN_MAP_RELATIVE_PATH
    examples_folder = output_folder / "Examples"
    metadata_folder = output_folder / "Metadata"

    map_path.parent.mkdir(parents=True, exist_ok=True)
    source_image.save(map_path)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    examples_folder.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, dict[str, list[Any]]] = {
        split: {"names": [], "queries": [], "database": []} for split in SPLIT_NAMES
    }

    for index, (x, y) in enumerate(centers):
        split = split_by_index[index]
        image_name = f"@{float(y):.3f}@{float(x):.3f}@synthetic_geotiff@{index:06d}.png"

        query_chip = crop_center(source_image, x, y, query_size)
        query_image = make_synthetic_query(query_chip)
        database_chip = crop_center(source_image, x, y, query_size)

        split_rows[split]["names"].append(image_name)
        split_rows[split]["queries"].append(np.asarray(query_image, dtype=np.uint8))
        split_rows[split]["database"].append(np.asarray(database_chip, dtype=np.uint8))

        if index == 0:
            crop_center(source_image, x, y, database_size).save(examples_folder / "satellite_database_crop.png")
            query_image.save(examples_folder / "synthetic_thermal_query.png")

    for split in SPLIT_NAMES:
        names = list(split_rows[split]["names"])
        write_sthn_h5(
            dataset_folder / f"{split}_queries.h5",
            image_names=names,
            images=list(split_rows[split]["queries"]),
            image_size=query_size,
            compression=compression,
        )
        write_sthn_h5(
            dataset_folder / f"{split}_database.h5",
            image_names=names,
            images=list(split_rows[split]["database"]),
            image_size=query_size,
            compression=compression,
        )

    metadata_path = metadata_folder / "sthn_export_metadata.json"
    readme_path = output_folder / "README.md"
    split_summary = {split: len(split_rows[split]["names"]) for split in SPLIT_NAMES}
    payload = {
        "format": "STHN loader-compatible",
        "source_geotiff": str(geotiff_path),
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "map_path": str(map_path),
        "database_size": database_size,
        "query_size": query_size,
        "stride": effective_stride,
        "sample_count": len(centers),
        "splits": split_summary,
        "h5_keys": ["image_data", "image_size", "image_name"],
        "warning": (
            "Query images are synthetic grayscale chips derived from the satellite raster. "
            "They make the pipeline loadable but are not real thermal observations."
        ),
        "recommended_loader_args": {
            "datasets_folder": str(dataset_root),
            "dataset_name": dataset_name,
            "split": "test",
            "database_size": database_size,
            "crop_width": query_size,
            "resize_width": 256,
            "batch_size": 1,
            "num_workers": 0,
        },
    }
    write_json(metadata_path, payload)
    write_readme(readme_path, dataset_root, dataset_name, database_size, query_size)

    LOGGER.info("Wrote STHN model input: %s", dataset_root)
    return STHNExportResult(
        output_folder=output_folder,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        map_path=map_path,
        metadata_path=metadata_path,
        readme_path=readme_path,
        sample_count=len(centers),
    )

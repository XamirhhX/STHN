from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


MAP_REL = Path("maps") / "satellite" / "20201117_BingSatellite.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an overlapping STHN map-crop dataset H5 from a large preview PNG."
    )
    parser.add_argument("--map_path", required=True)
    parser.add_argument("--datasets_folder", required=True)
    parser.add_argument("--dataset_name", default="sthn_overlap_tiles")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tile_size_px", type=int, default=1536)
    parser.add_argument("--tile_stride_px", type=int, default=512)
    parser.add_argument("--meters_per_pixel", type=float, default=5.0)
    parser.add_argument("--copy_mode", default="hardlink", choices=["hardlink", "copy", "none"])
    return parser.parse_args()


def covering_centers(size_px: int, tile_size_px: int, stride_px: int) -> list[int]:
    if tile_size_px > size_px:
        raise ValueError("tile_size_px cannot exceed map size")
    half = tile_size_px // 2
    centers = list(range(half, size_px - half + 1, stride_px))
    final_center = size_px - half
    if centers[-1] != final_center:
        centers.append(final_center)
    return centers


def ensure_map(args: argparse.Namespace, map_path: Path) -> Path:
    output_map = Path(args.datasets_folder) / MAP_REL
    output_map.parent.mkdir(parents=True, exist_ok=True)
    if args.copy_mode == "none":
        return map_path
    if output_map.exists():
        output_map.unlink()
    if args.copy_mode == "hardlink":
        try:
            os.link(map_path, output_map)
            return output_map
        except OSError:
            pass
    shutil.copy2(map_path, output_map)
    return output_map


def main() -> None:
    args = parse_args()
    map_path = Path(args.map_path).resolve()
    if not map_path.exists():
        raise FileNotFoundError(map_path)
    if args.tile_size_px <= 0 or args.tile_stride_px <= 0:
        raise ValueError("tile size and stride must be positive")

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(map_path) as image:
        width, height = image.size
    if width != height:
        raise ValueError(f"Expected square map preview, got {width} x {height}")

    centers_y = covering_centers(height, args.tile_size_px, args.tile_stride_px)
    centers_x = covering_centers(width, args.tile_size_px, args.tile_stride_px)
    records = []
    for row, y in enumerate(centers_y):
        for col, x in enumerate(centers_x):
            records.append(
                {
                    "name": f"@{y}@{x}@sthn_overlap@r{row:02d}@c{col:02d}.png",
                    "y": y,
                    "x": x,
                    "row": row,
                    "col": col,
                }
            )

    dataset_dir = Path(args.datasets_folder) / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    h5_path = dataset_dir / f"{args.split}_database.h5"
    if h5_path.exists():
        h5_path.unlink()

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["source"] = "map_crop_overlap"
        h5.attrs["map_path"] = str(ensure_map(args, map_path))
        h5.attrs["map_width_px"] = width
        h5.attrs["map_height_px"] = height
        h5.attrs["tile_size_px"] = args.tile_size_px
        h5.attrs["tile_stride_px"] = args.tile_stride_px
        h5.attrs["meters_per_pixel"] = args.meters_per_pixel
        h5.attrs["tile_size_m"] = args.tile_size_px * args.meters_per_pixel
        h5.attrs["tile_stride_m"] = args.tile_stride_px * args.meters_per_pixel
        h5.create_dataset(
            "image_name",
            data=[record["name"] for record in records],
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="lzf",
        )
        h5.create_dataset(
            "image_size",
            data=np.asarray([[args.tile_size_px, args.tile_size_px]] * len(records), dtype=np.int32),
            compression="lzf",
        )

    manifest = {
        "map_path": str(map_path),
        "dataset_h5": str(h5_path),
        "count": len(records),
        "rows": len(centers_y),
        "cols": len(centers_x),
        "tile_size_px": args.tile_size_px,
        "tile_stride_px": args.tile_stride_px,
        "meters_per_pixel": args.meters_per_pixel,
        "tile_size_m": args.tile_size_px * args.meters_per_pixel,
        "tile_stride_m": args.tile_stride_px * args.meters_per_pixel,
    }
    (dataset_dir / f"{args.split}_overlap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} overlapping crop records to {h5_path}")
    print(f"Grid: {len(centers_y)} x {len(centers_x)}")


if __name__ == "__main__":
    main()

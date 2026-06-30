from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create professor-scale 50 km retrieval H5 files from a GeoTIFF/PNG map."
    )
    parser.add_argument("--map_path", required=True, help="Source RGB map image exported from GeoTIFF.")
    parser.add_argument("--datasets_folder", required=True)
    parser.add_argument("--dataset_name", default="retrieval_iran_lut_50km_two_grid")
    parser.add_argument("--split", default="test")
    parser.add_argument("--map_size_m", type=float, default=52000.0)
    parser.add_argument("--search_size_m", type=float, default=50000.0)
    parser.add_argument("--meters_per_pixel", type=float, default=2.0)
    parser.add_argument("--grid_size", type=int, default=25)
    parser.add_argument("--tile_period_m", type=float, default=2000.0)
    parser.add_argument("--database_crop_m", type=float, default=2000.0)
    parser.add_argument("--query_crop_m", type=float, default=1024.0)
    parser.add_argument("--output_size_px", type=int, default=512)
    parser.add_argument("--zoom_level", type=int, default=16)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--database", action="store_true", help="Write *_database.h5.")
    parser.add_argument("--queries", action="store_true", help="Write *_queries.h5.")
    parser.add_argument("--query_grayscale", action="store_true", help="Store grayscale-derived RGB query chips.")
    return parser.parse_args()


def compression_value(raw: str) -> str | None:
    return None if raw == "none" else raw


def crop_resize(image: Image.Image, center_px: tuple[float, float], crop_px: int, output_size_px: int) -> Image.Image:
    half = crop_px / 2.0
    center_y, center_x = center_px
    box = (
        int(round(center_x - half)),
        int(round(center_y - half)),
        int(round(center_x + half)),
        int(round(center_y + half)),
    )
    tile = image.crop(box)
    if tile.size != (crop_px, crop_px):
        raise ValueError(f"Crop box {box} produced {tile.size}, expected {(crop_px, crop_px)}")
    if crop_px != output_size_px:
        tile = tile.resize((output_size_px, output_size_px), Image.Resampling.LANCZOS)
    return tile.convert("RGB")


def metric_to_pixel(y_m: float, x_m: float, margin_m: float, meters_per_pixel: float) -> tuple[float, float]:
    return ((margin_m + y_m) / meters_per_pixel, (margin_m + x_m) / meters_per_pixel)


def database_records(args: argparse.Namespace) -> list[dict]:
    records: list[dict] = []
    for grid_id, shift in [(0, 0.0), (1, 0.5)]:
        for row in range(args.grid_size):
            for col in range(args.grid_size):
                y_m = (0.5 + row + shift) * args.tile_period_m
                x_m = (0.5 + col + shift) * args.tile_period_m
                records.append(
                    {
                        "name": f"@{y_m:.3f}@{x_m:.3f}@grid{grid_id}@z{args.zoom_level}@r{row:02d}@c{col:02d}.png",
                        "y_m": y_m,
                        "x_m": x_m,
                        "row": row,
                        "col": col,
                        "grid_id": grid_id,
                    }
                )
    return records


def query_records(args: argparse.Namespace) -> list[dict]:
    records: list[dict] = []
    for row in range(args.grid_size):
        for col in range(args.grid_size):
            y_m = (0.5 + row) * args.tile_period_m
            x_m = (0.5 + col) * args.tile_period_m
            records.append(
                {
                    "name": f"@{y_m:.3f}@{x_m:.3f}@synthetic_query@z{args.zoom_level}@r{row:02d}@c{col:02d}.png",
                    "y_m": y_m,
                    "x_m": x_m,
                    "row": row,
                    "col": col,
                }
            )
    return records


def validate_args(args: argparse.Namespace, image: Image.Image) -> None:
    if not args.database and not args.queries:
        args.database = True
        args.queries = True
    if args.map_size_m <= args.search_size_m:
        raise ValueError("--map_size_m must exceed --search_size_m to leave shifted-grid margin.")
    if args.meters_per_pixel <= 0:
        raise ValueError("--meters_per_pixel must be positive.")
    if args.output_size_px <= 0:
        raise ValueError("--output_size_px must be positive.")
    if args.grid_size <= 0:
        raise ValueError("--grid_size must be positive.")
    if args.tile_period_m <= 0:
        raise ValueError("--tile_period_m must be positive.")

    expected_px = int(round(args.map_size_m / args.meters_per_pixel))
    if image.size != (expected_px, expected_px):
        raise ValueError(
            f"Map image is {image.size}, expected {(expected_px, expected_px)} from "
            f"map_size_m/meters_per_pixel."
        )
    if args.grid_size * args.tile_period_m != args.search_size_m:
        raise ValueError("grid_size * tile_period_m must equal search_size_m.")


def write_h5(
    path: Path,
    image: Image.Image,
    records: list[dict],
    crop_m: float,
    args: argparse.Namespace,
    grayscale: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    margin_m = (args.map_size_m - args.search_size_m) / 2.0
    crop_px = int(round(crop_m / args.meters_per_pixel))
    compression = compression_value(args.compression)
    names = [record["name"] for record in records]

    with h5py.File(path, "w") as h5:
        h5.attrs["map_size_m"] = args.map_size_m
        h5.attrs["search_size_m"] = args.search_size_m
        h5.attrs["meters_per_pixel"] = args.meters_per_pixel
        h5.attrs["grid_size"] = args.grid_size
        h5.attrs["tile_period_m"] = args.tile_period_m
        h5.attrs["crop_m"] = crop_m
        h5.attrs["output_size_px"] = args.output_size_px
        h5.attrs["zoom_level"] = args.zoom_level

        data = h5.create_dataset(
            "image_data",
            shape=(len(records), args.output_size_px, args.output_size_px, 3),
            dtype=np.uint8,
            chunks=(1, args.output_size_px, args.output_size_px, 3),
            compression=compression,
        )
        h5.create_dataset(
            "image_size",
            data=np.asarray([[args.output_size_px, args.output_size_px]] * len(records), dtype=np.int32),
            compression=compression,
        )
        h5.create_dataset("image_name", data=names, dtype=h5py.string_dtype(encoding="utf-8"), compression=compression)

        for index, record in enumerate(tqdm(records, desc=f"Writing {path.name}")):
            center_px = metric_to_pixel(record["y_m"], record["x_m"], margin_m, args.meters_per_pixel)
            tile = crop_resize(image, center_px, crop_px, args.output_size_px)
            if grayscale:
                tile = ImageOps.grayscale(tile).convert("RGB")
            data[index] = np.asarray(tile, dtype=np.uint8)


def write_manifest(path: Path, args: argparse.Namespace, database_count: int, query_count: int) -> None:
    payload = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "map_path": str(Path(args.map_path).resolve()),
        "map_size_m": args.map_size_m,
        "search_size_m": args.search_size_m,
        "meters_per_pixel": args.meters_per_pixel,
        "grid_size": args.grid_size,
        "tile_period_m": args.tile_period_m,
        "database_crop_m": args.database_crop_m,
        "query_crop_m": args.query_crop_m,
        "output_size_px": args.output_size_px,
        "zoom_level": args.zoom_level,
        "database_tiles": database_count,
        "query_tiles": query_count,
        "note": "50 km search region plus 1 km margin on each side for the half-shifted grid.",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(args.map_path).convert("RGB")
    validate_args(args, image)

    dataset_dir = Path(args.datasets_folder) / args.dataset_name
    db_records = database_records(args)
    q_records = query_records(args)

    if args.database:
        write_h5(dataset_dir / f"{args.split}_database.h5", image, db_records, args.database_crop_m, args)
    if args.queries:
        write_h5(
            dataset_dir / f"{args.split}_queries.h5",
            image,
            q_records,
            args.query_crop_m,
            args,
            grayscale=args.query_grayscale,
        )
    write_manifest(dataset_dir / f"{args.split}_professor_scale_manifest.json", args, len(db_records), len(q_records))
    print(f"Wrote professor-scale retrieval dataset to {dataset_dir}")
    print(f"Database records: {len(db_records)}")
    print(f"Query records: {len(q_records)}")


if __name__ == "__main__":
    main()

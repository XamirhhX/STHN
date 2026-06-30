from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_MAP_REL = Path("maps") / "satellite" / "20201117_BingSatellite.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a two-grid satellite tile database for spiral retrieval."
    )
    parser.add_argument("--map_path", required=True, help="Large satellite map image.")
    parser.add_argument("--datasets_folder", default="datasets")
    parser.add_argument("--dataset_name", default="retrieval_level16_two_grid")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "extended"])
    parser.add_argument("--center_y", type=float, default=None, help="Prior center row/pixel. Defaults to map center.")
    parser.add_argument("--center_x", type=float, default=None, help="Prior center col/pixel. Defaults to map center.")
    parser.add_argument("--grid_size", type=int, default=25, help="Tiles per side in each grid.")
    parser.add_argument("--tile_size_px", type=int, default=512)
    parser.add_argument("--tile_stride_px", type=int, default=None)
    parser.add_argument("--meters_per_pixel", type=float, default=2.0)
    parser.add_argument("--zoom_level", type=int, default=16)
    parser.add_argument("--grids", default="both", choices=["base", "shifted", "both"])
    parser.add_argument("--include_image_data", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--export_tiles_dir", default=None)
    parser.add_argument(
        "--copy_map",
        action="store_true",
        help="Copy map to datasets/maps/satellite/20201117_BingSatellite.png for BaseDataset.",
    )
    return parser.parse_args()


def selected_grids(grids: str) -> list[tuple[int, float, float]]:
    if grids == "base":
        return [(0, 0.0, 0.0)]
    if grids == "shifted":
        return [(1, 0.5, 0.5)]
    return [(0, 0.0, 0.0), (1, 0.5, 0.5)]


def valid_crop(center_y: int, center_x: int, tile_size_px: int, height: int, width: int) -> bool:
    half = tile_size_px // 2
    return (
        center_y - half >= 0
        and center_x - half >= 0
        and center_y + half <= height
        and center_x + half <= width
    )


def crop_tile(image: Image.Image, center_y: int, center_x: int, tile_size_px: int) -> Image.Image:
    half = tile_size_px // 2
    return image.crop((center_x - half, center_y - half, center_x + half, center_y + half))


def build_centers(args: argparse.Namespace, height: int, width: int) -> list[dict]:
    stride = args.tile_stride_px or args.tile_size_px
    center_y = args.center_y if args.center_y is not None else height / 2.0
    center_x = args.center_x if args.center_x is not None else width / 2.0
    mid = (args.grid_size - 1) / 2.0

    centers: list[dict] = []
    skipped = 0
    for grid_id, shift_y, shift_x in selected_grids(args.grids):
        for row in range(args.grid_size):
            for col in range(args.grid_size):
                y = int(round(center_y + (row - mid + shift_y) * stride))
                x = int(round(center_x + (col - mid + shift_x) * stride))
                if not valid_crop(y, x, args.tile_size_px, height, width):
                    skipped += 1
                    continue
                centers.append(
                    {
                        "grid_id": grid_id,
                        "row": row,
                        "col": col,
                        "y": y,
                        "x": x,
                        "name": f"@{y}@{x}@grid{grid_id}@z{args.zoom_level}",
                    }
                )
    if skipped:
        print(f"Skipped {skipped} tiles that would exceed map boundaries.")
    if not centers:
        raise RuntimeError("No valid tile centers were generated. Check center, grid size, and map dimensions.")
    return centers


def write_h5(args: argparse.Namespace, image: Image.Image, centers: list[dict]) -> Path:
    output_dir = Path(args.datasets_folder) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_database.h5"
    if output_path.exists():
        output_path.unlink()

    compression = "lzf" if args.compress else None
    names = [center["name"] for center in centers]
    image_sizes = np.asarray([[args.tile_size_px, args.tile_size_px]] * len(centers), dtype=np.int32)

    with h5py.File(output_path, "w") as h5:
        h5.attrs["zoom_level"] = args.zoom_level
        h5.attrs["meters_per_pixel"] = args.meters_per_pixel
        h5.attrs["tile_size_px"] = args.tile_size_px
        h5.attrs["tile_stride_px"] = args.tile_stride_px or args.tile_size_px
        h5.attrs["tile_size_m"] = args.tile_size_px * args.meters_per_pixel
        h5.attrs["grid_size"] = args.grid_size
        h5.attrs["grids"] = args.grids

        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("image_name", data=names, dtype=string_dtype, compression=compression)
        h5.create_dataset("image_size", data=image_sizes, compression=compression)

        if args.include_image_data:
            data = h5.create_dataset(
                "image_data",
                shape=(len(centers), args.tile_size_px, args.tile_size_px, 3),
                dtype=np.uint8,
                chunks=(1, args.tile_size_px, args.tile_size_px, 3),
                compression=compression,
            )
            for index, center in enumerate(tqdm(centers, desc="Writing image_data")):
                tile = crop_tile(image, center["y"], center["x"], args.tile_size_px)
                data[index] = np.asarray(tile.convert("RGB"), dtype=np.uint8)

    return output_path


def export_tiles(args: argparse.Namespace, image: Image.Image, centers: list[dict]) -> None:
    if not args.export_tiles_dir:
        return
    export_dir = Path(args.export_tiles_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for center in tqdm(centers, desc="Exporting tiles"):
        tile = crop_tile(image, center["y"], center["x"], args.tile_size_px)
        filename = (
            f"grid{center['grid_id']}_r{center['row']:02d}_c{center['col']:02d}"
            f"_y{center['y']}_x{center['x']}.png"
        )
        tile.save(export_dir / filename)


def maybe_copy_map(args: argparse.Namespace, map_path: Path) -> None:
    if not args.copy_map:
        return
    output_map = Path(args.datasets_folder) / DEFAULT_MAP_REL
    output_map.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(map_path, output_map)
    print(f"Copied map to {output_map}")


def main() -> None:
    args = parse_args()
    map_path = Path(args.map_path)
    if not map_path.exists():
        raise FileNotFoundError(map_path)
    if args.grid_size <= 0:
        raise ValueError("--grid_size must be positive")
    if args.tile_size_px <= 0:
        raise ValueError("--tile_size_px must be positive")
    if args.tile_stride_px is not None and args.tile_stride_px <= 0:
        raise ValueError("--tile_stride_px must be positive")
    if args.meters_per_pixel <= 0:
        raise ValueError("--meters_per_pixel must be positive")

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(map_path).convert("RGB")
    width, height = image.size

    centers = build_centers(args, height=height, width=width)
    output_path = write_h5(args, image=image, centers=centers)
    export_tiles(args, image=image, centers=centers)
    maybe_copy_map(args, map_path)

    print(f"Wrote {len(centers)} tile records to {output_path}")
    print(f"Tiles per requested grid: {args.grid_size} x {args.grid_size}")
    print(f"Nominal tile size: {args.tile_size_px * args.meters_per_pixel:.1f} m")


if __name__ == "__main__":
    main()

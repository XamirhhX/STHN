"""
Create a small test-only STHN dataset subset from a full extracted dataset.

This is intended for the practical case where the official dataset archive is too
large for a student's Drive. Run this on a machine/account that already has the
full dataset extracted, then zip/upload the generated subset.

The local test loader only needs:
  - maps/satellite/20201117_BingSatellite.png
  - <dataset_name>/<split>_queries.h5 with selected query images
  - <dataset_name>/<split>_database.h5 with database image names

Example:
    python scripts/create_minimal_sthn_subset.py ^
      --source_datasets /data/STHN/datasets ^
      --output_datasets /tmp/STHN_minimal_100 ^
      --dataset_name satellite_0_thermalmapping_135_train ^
      --split test ^
      --num_queries 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


MAP_REL = Path("maps") / "satellite" / "20201117_BingSatellite.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact STHN test subset.")
    parser.add_argument("--source_datasets", required=True, help="Full extracted datasets root.")
    parser.add_argument("--output_datasets", required=True, help="Destination minimal datasets root.")
    parser.add_argument("--dataset_name", default="satellite_0_thermalmapping_135_train")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--positive_radius", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selection",
        default="even",
        choices=["even", "first", "random"],
        help="How to choose query rows that have at least one database positive.",
    )
    return parser.parse_args()


def decode_name(value: bytes | str) -> str:
    return value.decode("UTF-8") if isinstance(value, bytes) else str(value)


def parse_utm(name: str) -> tuple[float, float]:
    parts = name.split("@")
    return float(parts[1]), float(parts[2])


def read_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as h5:
        return [decode_name(x) for x in h5["image_name"][:]]


def select_query_indexes(
    query_names: list[str],
    database_names: list[str],
    num_queries: int,
    positive_radius: float,
    selection: str,
    seed: int,
) -> np.ndarray:
    query_utms = np.array([parse_utm(name) for name in query_names], dtype=np.float64)
    database_utms = np.array([parse_utm(name) for name in database_names], dtype=np.float64)
    radius_sq = positive_radius**2

    valid = []
    for idx, q in enumerate(query_utms):
        # Chunking keeps memory bounded even if the database is large.
        has_positive = False
        for start in range(0, len(database_utms), 100_000):
            db = database_utms[start : start + 100_000]
            dist_sq = np.sum((db - q) ** 2, axis=1)
            if np.any(dist_sq <= radius_sq):
                has_positive = True
                break
        if has_positive:
            valid.append(idx)

    if len(valid) < num_queries:
        raise RuntimeError(
            f"Only found {len(valid)} queries with positives within {positive_radius} m; "
            f"requested {num_queries}."
        )

    valid_arr = np.array(valid, dtype=np.int64)
    if selection == "first":
        chosen = valid_arr[:num_queries]
    elif selection == "random":
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(valid_arr, size=num_queries, replace=False))
    elif selection == "even":
        positions = np.linspace(0, len(valid_arr) - 1, num_queries).round().astype(np.int64)
        chosen = valid_arr[positions]
    else:
        raise ValueError(selection)
    return np.sort(chosen)


def copy_query_h5(source_path: Path, dest_path: Path, selected: np.ndarray) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as src, h5py.File(dest_path, "w") as dst:
        total_rows = len(src["image_name"])
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for key in src.keys():
            data = src[key]
            if isinstance(data, h5py.Dataset) and data.shape and data.shape[0] == total_rows:
                dst.create_dataset(
                    key,
                    data=data[selected],
                    compression=data.compression,
                    compression_opts=data.compression_opts,
                )
            elif isinstance(data, h5py.Dataset):
                dst.create_dataset(key, data=data[()])
            else:
                src.copy(key, dst)


def copy_database_names_only(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as src, h5py.File(dest_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.create_dataset("image_name", data=src["image_name"][:])


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_datasets)
    output_root = Path(args.output_datasets)
    source_dataset = source_root / args.dataset_name
    output_dataset = output_root / args.dataset_name

    query_h5 = source_dataset / f"{args.split}_queries.h5"
    database_h5 = source_dataset / f"{args.split}_database.h5"
    source_map = source_root / MAP_REL

    for path in [query_h5, database_h5, source_map]:
        if not path.exists():
            raise FileNotFoundError(path)

    query_names = read_names(query_h5)
    database_names = read_names(database_h5)
    selected = select_query_indexes(
        query_names,
        database_names,
        args.num_queries,
        args.positive_radius,
        args.selection,
        args.seed,
    )

    output_map = output_root / MAP_REL
    output_map.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_map, output_map)

    copy_query_h5(query_h5, output_dataset / f"{args.split}_queries.h5", selected)
    copy_database_names_only(database_h5, output_dataset / f"{args.split}_database.h5")

    print(f"Wrote minimal dataset to: {output_root}")
    print(f"Selected queries: {len(selected)}")
    print(f"Query H5: {output_dataset / f'{args.split}_queries.h5'}")
    print(f"Database H5: {output_dataset / f'{args.split}_database.h5'}")
    print(f"Map: {output_map}")
    print("Use this as --datasets_folder in experiments/missing_data_eval.py")


if __name__ == "__main__":
    main()

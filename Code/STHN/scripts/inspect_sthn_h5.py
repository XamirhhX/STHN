"""
Inspect STHN H5 files and optionally preview a query image.

Example:
    python scripts/inspect_sthn_h5.py --h5 datasets/satellite_0_thermalmapping_135_train/test_queries.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an STHN .h5 file.")
    parser.add_argument("--h5", required=True, help="Path to an H5 file.")
    parser.add_argument("--num_names", type=int, default=5)
    return parser.parse_args()


def decode(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    args = parse_args()
    path = Path(args.h5)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5:
        print(f"H5: {path}")
        print(f"Keys: {list(h5.keys())}")
        print()

        for key in h5.keys():
            obj = h5[key]
            if isinstance(obj, h5py.Dataset):
                print(f"{key}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{key}: {type(obj)}")

        if "image_name" in h5:
            print()
            print(f"Rows: {len(h5['image_name'])}")
            print("First image names:")
            for value in h5["image_name"][: args.num_names]:
                name = decode(value)
                print(f"  {name}")
                parts = name.split("@")
                if len(parts) > 2:
                    try:
                        print(f"    parsed coordinate: easting={float(parts[1])}, northing={float(parts[2])}")
                    except ValueError:
                        pass

        if "image_data" in h5:
            data = h5["image_data"]
            print()
            print("image_data quick stats:")
            print(f"  shape: {data.shape}")
            print(f"  dtype: {data.dtype}")
            sample = data[0]
            print(f"  first sample shape: {sample.shape}")
            print(f"  first sample min/max: {sample.min()} / {sample.max()}")


if __name__ == "__main__":
    main()

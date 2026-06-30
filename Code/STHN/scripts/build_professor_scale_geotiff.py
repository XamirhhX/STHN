from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 52 km Level-16 GeoTIFF for professor-scale retrieval.")
    parser.add_argument(
        "--geotiff_project",
        default=r"C:\Users\Reacher\Documents\University\Project\01_Code\GeoTIFF",
        help="Path to the GeoTIFF project folder.",
    )
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--center_lat", type=float, default=30.886021)
    parser.add_argument("--center_lon", type=float, default=57.895202)
    parser.add_argument("--area_size_m", type=float, default=52000.0)
    parser.add_argument("--zoom", type=int, default=16)
    parser.add_argument("--output_size_px", type=int, default=26000)
    parser.add_argument("--max_tiles", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geotiff_project = Path(args.geotiff_project).resolve()
    sys.path.insert(0, str(geotiff_project))

    from Scripts import geo
    from Scripts.geotiff import build_geotiff

    # The GeoTIFF project intentionally defaults to 5 km demos. The professor
    # retrieval test needs 50 km plus a 1 km shifted-grid margin on each side.
    geo.MAX_AREA_SIZE_METERS = max(geo.MAX_AREA_SIZE_METERS, args.area_size_m)

    build_geotiff(
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        area_size_meters=args.area_size_m,
        zoom=args.zoom,
        output_folder=Path(args.output_folder),
        max_tiles=args.max_tiles,
        output_size_px=args.output_size_px,
        use_cache=True,
    )


if __name__ == "__main__":
    main()

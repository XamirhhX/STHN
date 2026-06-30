from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from .defaults import DEFAULT_SPLIT_RATIOS, DEFAULT_TILE_URL_TEMPLATE, DEFAULT_USER_AGENT
from .sthn_export import DEFAULT_STHN_DATASET_NAME
from .geo import validate_area_size, validate_center_lat_lon

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_common_geotiff_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--center-lat", type=float, required=True, help="Center latitude in WGS84 degrees.")
    parser.add_argument("--center-lon", type=float, required=True, help="Center longitude in WGS84 degrees.")
    parser.add_argument(
        "--area-size-meters",
        type=positive_float,
        default=500.0,
        help="Square area width/height in EPSG:3857 meters. Maximum: 5000. Default: 500.",
    )
    parser.add_argument("--zoom", type=non_negative_int, default=18, help="Tile zoom level. Default: 18.")
    parser.add_argument(
        "--output-size-px",
        type=positive_int,
        default=None,
        help="Optional square output size in pixels, e.g. 512 for STHN chips. Default: native tile resolution.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("Results/latest"),
        help="Output folder. Default: Results/latest.",
    )
    parser.add_argument(
        "--tile-url-template",
        default=DEFAULT_TILE_URL_TEMPLATE,
        help="Tile URL template with {z}, {x}, {y}, and optional {quadkey}.",
    )
    parser.add_argument("--retries", type=non_negative_int, default=3, help="Download retries per tile.")
    parser.add_argument("--timeout-seconds", type=positive_float, default=20.0, help="HTTP timeout per request.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent for tile requests.")
    parser.add_argument("--max-tiles", type=positive_int, default=256, help="Refuse downloads above this tile count.")
    parser.add_argument("--no-cache", action="store_true", help="Disable tile cache under the output folder.")


def add_patch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patch-size", type=positive_int, default=512, help="Patch size in pixels. Default: 512.")
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=None,
        help="Patch stride in pixels. Default: patch size.",
    )
    parser.add_argument(
        "--split-ratios",
        default=",".join(str(value) for value in DEFAULT_SPLIT_RATIOS),
        help="Train,val,test ratios. Default: 0.8,0.1,0.1.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic split seed.")


def add_sthn_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sthn-dataset-name",
        default=DEFAULT_STHN_DATASET_NAME,
        help=f"STHN dataset folder name. Default: {DEFAULT_STHN_DATASET_NAME}.",
    )
    parser.add_argument(
        "--sthn-database-size",
        type=positive_int,
        default=1536,
        help="Satellite crop size expected by the STHN loader. Default: 1536.",
    )
    parser.add_argument(
        "--sthn-query-size",
        type=positive_int,
        default=512,
        help="Synthetic query chip size stored in STHN H5 files. Default: 512.",
    )
    parser.add_argument(
        "--sthn-stride",
        type=positive_int,
        default=512,
        help="Sampling stride for STHN H5 centers. Default: 512.",
    )
    parser.add_argument(
        "--sthn-compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression for STHN image_data. Default: gzip.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geotiff-scripts",
        description="Create small GeoTIFFs and STHN patch datasets from map tiles.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    geotiff_parser = subparsers.add_parser("geotiff", help="Download tiles and write an organized GeoTIFF result.")
    add_common_geotiff_args(geotiff_parser)

    build_pipeline_parser = subparsers.add_parser("build", help="Run GeoTIFF, patch, and HDF5 export pipeline.")
    add_common_geotiff_args(build_pipeline_parser)
    add_patch_args(build_pipeline_parser)
    add_sthn_args(build_pipeline_parser)
    build_pipeline_parser.add_argument("--skip-patches", action="store_true", help="Only build the GeoTIFF.")
    build_pipeline_parser.add_argument("--skip-h5", action="store_true", help="Skip HDF5 export.")
    build_pipeline_parser.add_argument("--export-sthn", action="store_true", help="Also export STHN-compatible model input.")

    patches_parser = subparsers.add_parser("patches", help="Generate image patches from a GeoTIFF.")
    patches_parser.add_argument("--geotiff", type=Path, required=True, help="Input GeoTIFF path.")
    patches_parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("Patch_Dataset"),
        help="Patch dataset output folder. Default: Patch_Dataset.",
    )
    add_patch_args(patches_parser)

    h5_parser = subparsers.add_parser("export-h5", help="Export dataset manifests to HDF5.")
    h5_parser.add_argument("--dataset-folder", type=Path, required=True, help="Dataset folder from patches command.")
    h5_parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Folder for train.h5, val.h5, test.h5. Default: dataset folder.",
    )
    h5_parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 image compression. Default: gzip.",
    )

    sthn_parser = subparsers.add_parser("export-sthn", help="Export a GeoTIFF as STHN-compatible model input.")
    sthn_parser.add_argument("--geotiff", type=Path, required=True, help="Input GeoTIFF path.")
    sthn_parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("Results/latest/STHN_Model_Input"),
        help="Output folder for STHN model input. Default: Results/latest/STHN_Model_Input.",
    )
    sthn_parser.add_argument(
        "--split-ratios",
        default=",".join(str(value) for value in DEFAULT_SPLIT_RATIOS),
        help="Train,val,test ratios. Default: 0.8,0.1,0.1.",
    )
    sthn_parser.add_argument("--seed", type=int, default=1337, help="Deterministic split seed.")
    add_sthn_args(sthn_parser)

    settings_parser = subparsers.add_parser("from-settings", help="Run using setting.json.")
    settings_parser.add_argument(
        "--settings",
        type=Path,
        default=Path("setting.json"),
        help="Settings JSON file. Default: setting.json.",
    )
    settings_parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and tile estimate only.")
    settings_parser.add_argument("--center-lat", type=float, default=None, help="Override setting.json latitude.")
    settings_parser.add_argument("--center-lon", type=float, default=None, help="Override setting.json longitude.")
    settings_parser.add_argument("--area-size-meters", type=positive_float, default=None, help="Override square area size.")
    settings_parser.add_argument("--zoom", type=non_negative_int, default=None, help="Override imagery zoom.")
    settings_parser.add_argument("--output-size-px", type=positive_int, default=None, help="Override square output pixels.")
    settings_parser.add_argument("--output-folder", type=Path, default=None, help="Override output folder.")
    settings_parser.add_argument(
        "--reuse-output-folder",
        action="store_true",
        help="Write into the configured output folder even if it already contains files.",
    )
    return parser


def run_geotiff(args: argparse.Namespace) -> Path:
    from .geotiff import build_geotiff

    result = build_geotiff(
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        area_size_meters=args.area_size_meters,
        zoom=args.zoom,
        output_folder=args.output_folder,
        tile_url_template=args.tile_url_template,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        user_agent=args.user_agent,
        max_tiles=args.max_tiles,
        use_cache=not args.no_cache,
        output_size_px=args.output_size_px,
    )
    return result.geotiff_path


def run_patches(args: argparse.Namespace) -> Path:
    from .patches import generate_patches, parse_split_ratios

    split_ratios = parse_split_ratios(args.split_ratios)
    generate_patches(
        geotiff_path=args.geotiff,
        output_folder=args.output_folder,
        patch_size=args.patch_size,
        stride=args.stride,
        split_ratios=split_ratios,
        seed=args.seed,
    )
    return args.output_folder


def run_export_h5(args: argparse.Namespace) -> list[Path]:
    from .hdf5_export import export_h5

    compression = None if args.compression == "none" else args.compression
    return export_h5(
        dataset_folder=args.dataset_folder,
        output_folder=args.output_folder,
        compression=compression,
    )


def run_export_sthn(args: argparse.Namespace):
    from .patches import parse_split_ratios
    from .sthn_export import export_sthn_dataset

    compression = None if args.sthn_compression == "none" else args.sthn_compression
    return export_sthn_dataset(
        geotiff_path=args.geotiff,
        output_folder=args.output_folder,
        dataset_name=args.sthn_dataset_name,
        database_size=args.sthn_database_size,
        query_size=args.sthn_query_size,
        stride=args.sthn_stride,
        split_ratios=parse_split_ratios(args.split_ratios),
        seed=args.seed,
        compression=compression,
    )


def run_build(args: argparse.Namespace) -> None:
    geotiff_path = run_geotiff(args)
    if args.skip_patches:
        if args.export_sthn:
            sthn_args = argparse.Namespace(
                geotiff=geotiff_path,
                output_folder=args.output_folder / "STHN_Model_Input",
                sthn_dataset_name=args.sthn_dataset_name,
                sthn_database_size=args.sthn_database_size,
                sthn_query_size=args.sthn_query_size,
                sthn_stride=args.sthn_stride,
                sthn_compression=args.sthn_compression,
                split_ratios=args.split_ratios,
                seed=args.seed,
            )
            run_export_sthn(sthn_args)
        return
    dataset_folder = args.output_folder / "Patch_Dataset"
    patch_args = argparse.Namespace(
        geotiff=geotiff_path,
        output_folder=dataset_folder,
        patch_size=args.patch_size,
        stride=args.stride,
        split_ratios=args.split_ratios,
        seed=args.seed,
    )
    run_patches(patch_args)
    if args.skip_h5:
        if not args.export_sthn:
            return
    if not args.skip_h5:
        h5_args = argparse.Namespace(
            dataset_folder=dataset_folder,
            output_folder=dataset_folder,
            compression="gzip",
        )
        run_export_h5(h5_args)
    if args.export_sthn:
        sthn_args = argparse.Namespace(
            geotiff=geotiff_path,
            output_folder=args.output_folder / "STHN_Model_Input",
            sthn_dataset_name=args.sthn_dataset_name,
            sthn_database_size=args.sthn_database_size,
            sthn_query_size=args.sthn_query_size,
            sthn_stride=args.sthn_stride,
            sthn_compression=args.sthn_compression,
            split_ratios=args.split_ratios,
            seed=args.seed,
        )
        run_export_sthn(sthn_args)


def run_from_settings(args: argparse.Namespace) -> None:
    from .output_paths import unique_output_folder
    from .settings import estimate_tile_count, load_run_settings, run_settings, validate_run_settings

    settings = load_run_settings(args.settings)
    overrides = {}
    if args.center_lat is not None:
        overrides["center_lat"] = args.center_lat
    if args.center_lon is not None:
        overrides["center_lon"] = args.center_lon
    if args.area_size_meters is not None:
        overrides["area_size_meters"] = args.area_size_meters
    if args.zoom is not None:
        overrides["zoom"] = args.zoom
    if args.output_size_px is not None:
        overrides["output_size_px"] = args.output_size_px
    if args.output_folder is not None:
        overrides["output_folder"] = args.output_folder.resolve()
    if args.reuse_output_folder:
        overrides["unique_output_folder"] = False
    if overrides:
        settings = replace(settings, **overrides)
        validate_center_lat_lon(settings.center_lat, settings.center_lon)
        validate_area_size(settings.area_size_meters)
        validate_run_settings(settings)

    tile_count = estimate_tile_count(settings)
    if tile_count > settings.max_tiles:
        raise ValueError(
            f"Settings require {tile_count} tiles, above download.max_tiles {settings.max_tiles}. "
            "Lower imagery.zoom, shrink area, or raise download.max_tiles intentionally."
        )

    if args.dry_run:
        payload = settings.to_json_dict()
        payload["estimated_tiles"] = tile_count
        effective_output = (
            unique_output_folder(settings.output_folder)
            if settings.unique_output_folder
            else settings.output_folder
        )
        payload["effective_output_folder"] = str(effective_output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    LOGGER.info("Using settings file: %s", args.settings)
    LOGGER.info("Estimated tiles: %d", tile_count)
    run_settings(settings)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        if args.command == "geotiff":
            run_geotiff(args)
        elif args.command == "patches":
            run_patches(args)
        elif args.command == "export-h5":
            run_export_h5(args)
        elif args.command == "export-sthn":
            run_export_sthn(args)
        elif args.command == "build":
            run_build(args)
        elif args.command == "from-settings":
            run_from_settings(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        LOGGER.error("%s", exc)
        if args.verbose:
            raise
        return 1
    return 0

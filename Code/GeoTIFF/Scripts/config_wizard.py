from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_TILE_URL_TEMPLATE, DEFAULT_USER_AGENT
from .geo import (
    MAX_AREA_SIZE_METERS,
    area_bounds_around_center,
    lonlat_to_web_mercator,
    validate_area_size,
    validate_center_lat_lon,
    web_mercator_to_lonlat,
)
from .tiles import required_tiles

STREET_TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


@dataclass(frozen=True)
class Preset:
    slug: str
    label: str
    center_lat: float
    center_lon: float
    default_area_m: float
    default_zoom: int
    default_radius_m: float


PRESETS: tuple[Preset, ...] = (
    Preset("namib-desert", "Namib Desert dunes, Namibia", -24.7390, 15.2880, 5000.0, 16, 15_000.0),
    Preset("sahara-desert", "Sahara desert, Egypt", 23.4162, 25.6628, 5000.0, 16, 30_000.0),
    Preset("tehran-city", "Dense city, Tehran, Iran", 35.6892, 51.3890, 500.0, 18, 2_000.0),
    Preset("amazon-forest", "Amazon rainforest, Brazil", -3.4653, -62.2159, 1000.0, 17, 15_000.0),
    Preset("baja-coast", "Baja California coast, Mexico", 24.1426, -110.3128, 1500.0, 17, 8_000.0),
    Preset("swiss-alps", "Swiss Alps, Switzerland", 46.8523, 9.5320, 2000.0, 17, 8_000.0),
    Preset("california-farms", "Farm fields, California", 36.6026, -119.5108, 2000.0, 17, 10_000.0),
    Preset("greenland-ice", "Ice sheet, Greenland", 72.0000, -40.0000, 5000.0, 16, 25_000.0),
)


def prompt_text(label: str, default: str | None = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value or not required:
            return value
        print("Please enter a value.")


def prompt_float(
    label: str,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    default_text = None if default is None else f"{default:g}"
    while True:
        raw = prompt_text(label, default_text)
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum:g}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be at most {maximum:g}.")
            continue
        return value


def prompt_int(
    label: str,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    default_text = None if default is None else str(default)
    while True:
        raw = prompt_text(label, default_text)
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be at most {maximum}.")
            continue
        return value


def prompt_yes_no(label: str, default: bool = True) -> bool:
    default_text = "Y" if default else "N"
    while True:
        raw = prompt_text(f"{label} (Y/N)", default_text).lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter Y or N.")


def prompt_choice(title: str, options: list[tuple[str, str]], default: str) -> str:
    print()
    print(title)
    for key, label in options:
        marker = " default" if key == default else ""
        print(f"  {key}. {label}{marker}")
    valid = {key for key, _ in options}
    while True:
        value = prompt_text("Select", default)
        if value in valid:
            return value
        print(f"Choose one of: {', '.join(sorted(valid))}.")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "custom"


def auto_zoom_for_area(area_size_meters: float) -> int:
    if area_size_meters <= 750:
        return 18
    if area_size_meters <= 2500:
        return 17
    return 16


def choose_build_kind() -> str:
    return prompt_choice(
        "What do you want to create?",
        [
            ("1", "GeoTIFF + preview + metadata"),
            ("2", "GeoTIFF plus STHN PNG/JSON patches"),
            ("3", "Full STHN dataset: GeoTIFF + patches + train/val/test HDF5"),
        ],
        "1",
    )


def choose_location() -> tuple[str, float | None, float | None, float, int, float, dict[str, float] | None]:
    mode = prompt_choice(
        "What part of Earth should it come from?",
        [
            ("1", "Choose a preset Earth region"),
            ("2", "Use exact center latitude/longitude"),
            ("3", "Sample random centers inside a bounding box"),
        ],
        "1",
    )

    if mode == "1":
        options = [(str(index + 1), preset.label) for index, preset in enumerate(PRESETS)]
        selected = prompt_choice("Preset regions", options, "1")
        preset = PRESETS[int(selected) - 1]
        return (
            preset.slug,
            preset.center_lat,
            preset.center_lon,
            preset.default_area_m,
            preset.default_zoom,
            preset.default_radius_m,
            None,
        )

    if mode == "2":
        center_lat = prompt_float("Center latitude", minimum=-85.0, maximum=85.0)
        center_lon = prompt_float("Center longitude", minimum=-180.0, maximum=180.0)
        validate_center_lat_lon(center_lat, center_lon)
        place_name = prompt_text("Short place name for folders", "custom")
        return (slugify(place_name), center_lat, center_lon, 500.0, 18, 5_000.0, None)

    print()
    print("Enter a WGS84 bounding box. Centers will be sampled inside it.")
    west = prompt_float("West longitude", minimum=-180.0, maximum=180.0)
    south = prompt_float("South latitude", minimum=-85.0, maximum=85.0)
    east = prompt_float("East longitude", minimum=-180.0, maximum=180.0)
    north = prompt_float("North latitude", minimum=-85.0, maximum=85.0)
    if west >= east:
        raise ValueError("West longitude must be smaller than east longitude.")
    if south >= north:
        raise ValueError("South latitude must be smaller than north latitude.")
    place_name = prompt_text("Short region name for folders", "bbox-region")
    bbox = {"west": west, "south": south, "east": east, "north": north}
    return (slugify(place_name), None, None, 500.0, 18, 5_000.0, bbox)


def choose_area(default_area_m: float) -> float:
    print()
    print("Length means the width and height of each square map chip in meters.")
    area_size = prompt_float(
        "Chip length in meters",
        default=default_area_m,
        minimum=1.0,
        maximum=MAX_AREA_SIZE_METERS,
    )
    validate_area_size(area_size)
    return area_size


def choose_output_size_px() -> int | None:
    selected = prompt_choice(
        "What should the final GeoTIFF/image size be?",
        [
            ("1", "512 x 512 pixels, STHN standard"),
            ("2", "1024 x 1024 pixels"),
            ("3", "Custom square pixel size"),
            ("4", "Native tile resolution, pixels depend on length and zoom"),
        ],
        "1",
    )
    if selected == "1":
        return 512
    if selected == "2":
        return 1024
    if selected == "3":
        return prompt_int("Output image size in pixels", default=512, minimum=1)
    return None


def choose_zoom(default_zoom: int, area_size_meters: float) -> int:
    auto_zoom = auto_zoom_for_area(area_size_meters)
    selected = prompt_choice(
        "How close / detailed should the imagery be?",
        [
            ("1", f"Auto for this length: zoom {auto_zoom}"),
            ("2", "Very close: zoom 18, best for around 500 m"),
            ("3", "Close: zoom 17, good for around 1-2 km"),
            ("4", "Wide: zoom 16, good for around 2-5 km"),
            ("5", f"Preset/default: zoom {default_zoom}"),
            ("6", "Custom zoom"),
        ],
        "1",
    )
    if selected == "1":
        return auto_zoom
    if selected == "2":
        return 18
    if selected == "3":
        return 17
    if selected == "4":
        return 16
    if selected == "5":
        return default_zoom
    return prompt_int("Zoom level", default=auto_zoom, minimum=0, maximum=22)


def choose_layer() -> tuple[str, str]:
    selected = prompt_choice(
        "What imagery source should be used?",
        [
            ("1", "Satellite imagery, real aerial/satellite photos"),
            ("2", "Street map tiles, useful only for debug"),
            ("3", "Custom tile URL template"),
        ],
        "1",
    )
    if selected == "1":
        return "satellite", DEFAULT_TILE_URL_TEMPLATE
    if selected == "2":
        return "street", STREET_TILE_URL_TEMPLATE
    return "custom", prompt_text("Tile URL template with {z}, {x}, {y}")


def random_center_near(
    base_lat: float,
    base_lon: float,
    radius_meters: float,
    rng: random.Random,
) -> tuple[float, float]:
    base_x, base_y = lonlat_to_web_mercator(base_lon, base_lat)
    radius = radius_meters * math.sqrt(rng.random())
    angle = rng.uniform(0.0, math.tau)
    lon, lat = web_mercator_to_lonlat(base_x + radius * math.cos(angle), base_y + radius * math.sin(angle))
    validate_center_lat_lon(lat, lon)
    return lat, lon


def grid_centers(
    base_lat: float,
    base_lon: float,
    count: int,
    spacing_meters: float,
) -> list[tuple[float, float]]:
    base_x, base_y = lonlat_to_web_mercator(base_lon, base_lat)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    centers: list[tuple[float, float]] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        dx = (col - (cols - 1) / 2.0) * spacing_meters
        dy = ((rows - 1) / 2.0 - row) * spacing_meters
        lon, lat = web_mercator_to_lonlat(base_x + dx, base_y + dy)
        validate_center_lat_lon(lat, lon)
        centers.append((lat, lon))
    return centers


def choose_centers(
    count: int,
    center_lat: float | None,
    center_lon: float | None,
    bbox: dict[str, float] | None,
    area_size_meters: float,
    default_radius_m: float,
) -> tuple[list[tuple[float, float]], int | None]:
    if bbox is not None:
        seed = prompt_int("Random seed", default=1337, minimum=0)
        rng = random.Random(seed)
        centers = [
            (
                rng.uniform(bbox["south"], bbox["north"]),
                rng.uniform(bbox["west"], bbox["east"]),
            )
            for _ in range(count)
        ]
        for lat, lon in centers:
            validate_center_lat_lon(lat, lon)
        return centers, seed

    if center_lat is None or center_lon is None:
        raise ValueError("A center point is required unless a bounding box is provided.")
    if count == 1:
        return [(center_lat, center_lon)], None

    seed = prompt_int("Random seed", default=1337, minimum=0)
    rng = random.Random(seed)

    selected = prompt_choice(
        "How should multiple map chips be placed?",
        [
            ("1", "Random centers near the selected center"),
            ("2", "Grid around the selected center"),
            ("3", "Same center repeated"),
        ],
        "1",
    )
    if selected == "1":
        radius = prompt_float(
            "Random radius around center in meters",
            default=max(default_radius_m, area_size_meters),
            minimum=0.0,
        )
        return [random_center_near(center_lat, center_lon, radius, rng) for _ in range(count)], seed
    if selected == "2":
        spacing = prompt_float("Grid spacing in meters", default=area_size_meters, minimum=1.0)
        return grid_centers(center_lat, center_lon, count, spacing), seed
    return [(center_lat, center_lon) for _ in range(count)], seed


def estimate_tiles(centers: list[tuple[float, float]], area_size_meters: float, zoom: int) -> list[int]:
    counts: list[int] = []
    for lat, lon in centers:
        bounds = area_bounds_around_center(lat, lon, area_size_meters)
        counts.append(len(required_tiles(bounds, zoom).tiles))
    return counts


def choose_patch_options(build_kind: str, output_size_px: int | None) -> tuple[int | None, int | None, bool]:
    if build_kind == "1":
        return None, None, False
    while True:
        if output_size_px is None:
            patch_size_choice = prompt_choice(
                "Patch size",
                [
                    ("1", "512 x 512"),
                    ("2", "1024 x 1024"),
                    ("3", "Custom"),
                ],
                "1",
            )
            if patch_size_choice == "1":
                patch_size = 512
            elif patch_size_choice == "2":
                patch_size = 1024
            else:
                patch_size = prompt_int("Patch size in pixels", default=512, minimum=1)
        else:
            patch_size_choice = prompt_choice(
                "Patch size",
                [
                    ("1", f"Same as final chip: {output_size_px} x {output_size_px}"),
                    ("2", "512 x 512"),
                    ("3", "1024 x 1024"),
                    ("4", "Custom"),
                ],
                "1",
            )
            if patch_size_choice == "1":
                patch_size = output_size_px
            elif patch_size_choice == "2":
                patch_size = 512
            elif patch_size_choice == "3":
                patch_size = 1024
            else:
                patch_size = prompt_int("Patch size in pixels", default=output_size_px, minimum=1)
            if patch_size > output_size_px:
                print(f"Patch size must be <= final chip size ({output_size_px}px).")
                continue
        break
    stride_choice = prompt_choice(
        "Patch stride",
        [
            ("1", "Same as patch size, no overlap"),
            ("2", "Half patch size, 50 percent overlap"),
            ("3", "Custom stride"),
        ],
        "1",
    )
    if stride_choice == "1":
        stride = patch_size
    elif stride_choice == "2":
        stride = max(1, patch_size // 2)
    else:
        stride = prompt_int("Stride in pixels", default=patch_size, minimum=1)
    return patch_size, stride, build_kind == "3"


def output_folder_for_sample(output_root: Path, count: int, index: int) -> Path:
    if count == 1:
        return output_root
    return output_root / f"sample_{index + 1:03d}"


def build_command(
    python_exe: str,
    project_root: Path,
    build_kind: str,
    center_lat: float,
    center_lon: float,
    area_size_meters: float,
    zoom: int,
    output_folder: Path,
    tile_url_template: str,
    max_tiles: int,
    output_size_px: int | None,
    patch_size: int | None,
    stride: int | None,
    include_h5: bool,
) -> list[str]:
    if build_kind == "1":
        command = [
            python_exe,
            "-B",
            str(project_root / "main.py"),
            "geotiff",
            "--center-lat",
            f"{center_lat:.8f}",
            "--center-lon",
            f"{center_lon:.8f}",
            "--area-size-meters",
            f"{area_size_meters:g}",
            "--zoom",
            str(zoom),
            "--output-folder",
            str(output_folder),
            "--tile-url-template",
            tile_url_template,
            "--user-agent",
            DEFAULT_USER_AGENT,
            "--max-tiles",
            str(max_tiles),
        ]
        if output_size_px is not None:
            command.extend(["--output-size-px", str(output_size_px)])
        return command

    command = [
        python_exe,
        "-B",
        str(project_root / "main.py"),
        "build",
        "--center-lat",
        f"{center_lat:.8f}",
        "--center-lon",
        f"{center_lon:.8f}",
        "--area-size-meters",
        f"{area_size_meters:g}",
        "--zoom",
        str(zoom),
        "--output-folder",
        str(output_folder),
        "--tile-url-template",
        tile_url_template,
        "--user-agent",
        DEFAULT_USER_AGENT,
        "--max-tiles",
        str(max_tiles),
    ]
    if output_size_px is not None:
        command.extend(["--output-size-px", str(output_size_px)])
    if patch_size is not None:
        command.extend(["--patch-size", str(patch_size)])
    if stride is not None:
        command.extend(["--stride", str(stride)])
    if not include_h5:
        command.append("--skip-h5")
    return command


def command_preview(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def write_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_wizard(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    print()
    print("=" * 60)
    print(" GeoTIFF Scripts dataset wizard")
    print("=" * 60)

    build_kind = choose_build_kind()
    place_slug, center_lat, center_lon, default_area, default_zoom, default_radius, bbox = choose_location()
    count = prompt_int("How many separate map chips / GeoTIFFs", default=1, minimum=1, maximum=100)
    area_size_meters = choose_area(default_area)
    output_size_px = choose_output_size_px()
    zoom = choose_zoom(default_zoom, area_size_meters)
    centers, seed = choose_centers(count, center_lat, center_lon, bbox, area_size_meters, default_radius)
    layer_name, tile_url_template = choose_layer()
    patch_size, stride, include_h5 = choose_patch_options(build_kind, output_size_px)

    size_label = "native" if output_size_px is None else f"{output_size_px}px"
    default_output = Path("Results") / f"{place_slug}-{count}x-{int(area_size_meters)}m-{size_label}-{layer_name}-z{zoom}"
    output_root = Path(prompt_text("Output folder", str(default_output))).resolve()

    tile_counts = estimate_tiles(centers, area_size_meters, zoom)
    max_tile_count = max(tile_counts)
    total_tile_count = sum(tile_counts)
    max_tiles = 256
    if max_tile_count > max_tiles:
        print()
        print(f"Warning: one chip may need {max_tile_count} tiles, above the default max of {max_tiles}.")
        if not prompt_yes_no(f"Raise max tiles per chip to {max_tile_count}", default=False):
            print("Cancelled. Choose a wider zoom or a smaller chip length.")
            return 1
        max_tiles = max_tile_count
    if total_tile_count > 1000:
        print()
        print(f"Warning: this run may download about {total_tile_count} tiles in total.")
        if not prompt_yes_no("Continue", default=False):
            print("Cancelled.")
            return 1

    commands = [
        build_command(
            python_exe=python_exe,
            project_root=project_root,
            build_kind=build_kind,
            center_lat=lat,
            center_lon=lon,
            area_size_meters=area_size_meters,
            zoom=zoom,
            output_folder=output_folder_for_sample(output_root, count, index),
            tile_url_template=tile_url_template,
            max_tiles=max_tiles,
            output_size_px=output_size_px,
            patch_size=patch_size,
            stride=stride,
            include_h5=include_h5,
        )
        for index, (lat, lon) in enumerate(centers)
    ]

    run_config = {
        "build_kind": build_kind,
        "place": place_slug,
        "count": count,
        "area_size_meters": area_size_meters,
        "zoom": zoom,
        "output_size_px": output_size_px,
        "layer": layer_name,
        "tile_url_template": tile_url_template,
        "seed": seed,
        "patch_size": patch_size,
        "stride": stride,
        "include_h5": include_h5,
        "max_tiles_per_chip": max_tiles,
        "estimated_tiles_per_chip": tile_counts,
        "estimated_tiles_total": total_tile_count,
        "centers": [{"center_lat": lat, "center_lon": lon} for lat, lon in centers],
        "commands": [command_preview(command) for command in commands],
    }

    print()
    print("=" * 60)
    print(" Run summary")
    print("=" * 60)
    print(f"Create:       {'GeoTIFF only' if build_kind == '1' else 'GeoTIFF + patches' if build_kind == '2' else 'Full HDF5 dataset'}")
    print(f"Earth part:   {place_slug}")
    print(f"How many:     {count}")
    print(f"Length:       {area_size_meters:g} m")
    print(f"Output size:  {'native tile resolution' if output_size_px is None else str(output_size_px) + ' x ' + str(output_size_px) + ' px'}")
    if output_size_px is not None:
        print(f"Resolution:   {area_size_meters / output_size_px:g} m/px")
    print(f"Zoom:         {zoom}")
    print(f"Imagery:      {layer_name}")
    print(f"Output:       {output_root}")
    print(f"Tiles:        max {max_tile_count} per chip, about {total_tile_count} total")
    if patch_size is not None:
        print(f"Patches:      {patch_size}px, stride {stride}px")
    print()
    print("Centers:")
    for index, (lat, lon) in enumerate(centers[:10], start=1):
        print(f"  {index:03d}: {lat:.8f}, {lon:.8f}")
    if len(centers) > 10:
        print(f"  ... {len(centers) - 10} more")
    print()
    print("First command:")
    print(command_preview(commands[0]))

    write_run_config(output_root / "run_config.json", run_config)
    print()
    print(f"Wrote run config: {output_root / 'run_config.json'}")

    if args.dry_run:
        print("Dry run only. Nothing was downloaded.")
        return 0

    if not prompt_yes_no("Run now", default=True):
        print("Cancelled. The run_config.json file was still saved.")
        return 0

    for index, command in enumerate(commands, start=1):
        print()
        print("=" * 60)
        print(f" Running chip {index} of {count}")
        print("=" * 60)
        completed = subprocess.run(command, cwd=project_root, check=False)
        if completed.returncode != 0:
            print(f"Build failed for chip {index}.")
            return completed.returncode

    print()
    print("Done.")
    print(f"Outputs are in: {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive GeoTIFF Scripts dataset wizard.")
    parser.add_argument("--dry-run", action="store_true", help="Ask questions and write run_config.json, but do not run downloads.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_wizard(args)
    except KeyboardInterrupt:
        print()
        print("Cancelled.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path


EARTH_CIRCUMFERENCE_M = 40075016.68557849
DEFAULT_ESRI_TILE_URL_TEMPLATE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

AREA_PRESETS = {
    "desert": {
        "label": "lut_desert_hard",
        "center_lat": 30.886021,
        "center_lon": 57.895202,
        "description": "Hard desert default near the Lut/Kerman desert test region.",
    },
    "city": {
        "label": "tehran_city",
        "center_lat": 35.6892,
        "center_lon": 51.3890,
        "description": "Dense urban default centered on Tehran.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download satellite 50 km previews and create STHN-sized "
            "1536 x 1536 satellite tile H5 files."
        )
    )
    parser.add_argument(
        "--provider",
        default="google",
        choices=["google", "esri"],
        help="Satellite tile provider. Google requires an API key; Esri uses a public XYZ endpoint.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Google Maps API key. Defaults to GOOGLE_MAPS_API_KEY; only used with --provider google.",
    )
    parser.add_argument(
        "--tile_url_template",
        default=DEFAULT_ESRI_TILE_URL_TEMPLATE,
        help="XYZ tile URL template for --provider esri. Supports {z}, {x}, and {y}.",
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        default=["desert", "city"],
        choices=sorted(AREA_PRESETS),
        help="Preset areas to generate.",
    )
    parser.add_argument(
        "--custom_area",
        action="append",
        default=[],
        metavar="LABEL,LAT,LON",
        help="Extra area definition. Can be repeated, e.g. hard_city,35.7,51.4.",
    )
    parser.add_argument("--output_root", default="datasets/google_50km_previews")
    parser.add_argument("--area_size_m", type=float, default=50000.0)
    parser.add_argument("--preview_size_px", type=int, default=10000)
    parser.add_argument("--zoom", type=int, default=15)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--region", default="IR")
    parser.add_argument("--image_format", default="jpeg", choices=["jpeg", "png"])
    parser.add_argument("--request_sleep_s", type=float, default=0.0)
    parser.add_argument("--tile_workers", type=int, default=16)
    parser.add_argument("--tile_retries", type=int, default=3)
    parser.add_argument(
        "--max_source_tiles",
        type=int,
        default=6000,
        help="Safety limit per area before download starts.",
    )
    parser.add_argument(
        "--tile_cache_dir",
        default=None,
        help=(
            "Optional cache for raw Google tile responses. Review Google Maps "
            "Platform content storage rules before enabling persistent cache."
        ),
    )
    parser.add_argument("--skip_existing_preview", action="store_true")
    parser.add_argument("--write_h5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sthn_tile_size_px", type=int, default=1536)
    parser.add_argument("--sthn_tile_stride_px", type=int, default=1536)
    parser.add_argument("--split", default="test")
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--export_sthn_tiles", action="store_true")
    return parser.parse_args()


def require_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Google Maps API key. Pass --api_key or set GOOGLE_MAPS_API_KEY."
        )
    return api_key


def build_area_list(args: argparse.Namespace) -> list[dict]:
    areas = []
    for key in args.areas:
        preset = AREA_PRESETS[key]
        areas.append(
            {
                "key": key,
                "label": preset["label"],
                "center_lat": float(preset["center_lat"]),
                "center_lon": float(preset["center_lon"]),
                "description": preset["description"],
            }
        )
    for raw in args.custom_area:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError("--custom_area must use LABEL,LAT,LON")
        label, lat, lon = parts
        areas.append(
            {
                "key": label,
                "label": label,
                "center_lat": float(lat),
                "center_lon": float(lon),
                "description": "User-provided custom area.",
            }
        )
    return areas


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(url: str) -> tuple[bytes, str | None]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read(), response.headers.get("Content-Type")


def create_google_session(args: argparse.Namespace, api_key: str) -> dict:
    query = urllib.parse.urlencode({"key": api_key})
    url = f"https://tile.googleapis.com/v1/createSession?{query}"
    payload = {
        "mapType": "satellite",
        "language": args.language,
        "region": args.region,
        "imageFormat": args.image_format,
    }
    session = post_json(url, payload)
    if "session" not in session:
        raise RuntimeError(f"Google session response did not contain a session token: {session}")
    return session


def lat_lon_to_world_px(lat: float, lon: float, zoom: int, tile_size_px: int) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    sin_lat = math.sin(math.radians(lat))
    scale = tile_size_px * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return x, y


def meters_per_pixel(lat: float, zoom: int, tile_size_px: int) -> float:
    return (
        math.cos(math.radians(lat))
        * EARTH_CIRCUMFERENCE_M
        / float(tile_size_px * (2**zoom))
    )


def source_crop_box(
    center_lat: float,
    center_lon: float,
    area_size_m: float,
    zoom: int,
    tile_size_px: int,
) -> tuple[tuple[int, int, int, int], float]:
    center_x, center_y = lat_lon_to_world_px(center_lat, center_lon, zoom, tile_size_px)
    native_mpp = meters_per_pixel(center_lat, zoom, tile_size_px)
    half_px = area_size_m / (2.0 * native_mpp)
    left = int(round(center_x - half_px))
    top = int(round(center_y - half_px))
    right = int(round(center_x + half_px))
    bottom = int(round(center_y + half_px))
    return (left, top, right, bottom), native_mpp


def tile_range_for_box(box: tuple[int, int, int, int], tile_size_px: int) -> tuple[range, range]:
    left, top, right, bottom = box
    x_start = math.floor(left / tile_size_px)
    x_end = math.floor((right - 1) / tile_size_px)
    y_start = math.floor(top / tile_size_px)
    y_end = math.floor((bottom - 1) / tile_size_px)
    return range(x_start, x_end + 1), range(y_start, y_end + 1)


def tile_cache_path(
    cache_dir: Path,
    provider: str,
    zoom: int,
    x: int,
    y: int,
    image_format: str,
) -> Path:
    suffix = "jpg" if image_format == "jpeg" else "png"
    return cache_dir / provider / str(zoom) / str(x) / f"{y}.{suffix}"


def download_tile(
    provider: str,
    api_key: str | None,
    session_token: str | None,
    zoom: int,
    x: int,
    y: int,
    image_format: str,
    tile_url_template: str,
    cache_dir: Path | None,
    request_sleep_s: float,
    tile_retries: int,
) -> Image.Image:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    world_tiles = 2**zoom
    wrapped_x = x % world_tiles
    if y < 0 or y >= world_tiles:
        raise ValueError(f"Tile y={y} is outside Web Mercator limits for zoom {zoom}.")

    cache_path = (
        tile_cache_path(cache_dir, provider, zoom, wrapped_x, y, image_format)
        if cache_dir
        else None
    )
    if cache_path and cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    if provider == "google":
        if not api_key or not session_token:
            raise ValueError("Google tile downloads require api_key and session_token.")
        query = urllib.parse.urlencode({"session": session_token, "key": api_key})
        url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{wrapped_x}/{y}?{query}"
    else:
        url = tile_url_template.format(z=zoom, x=wrapped_x, y=y)
    data = None
    for attempt in range(tile_retries + 1):
        try:
            data, _ = get_bytes(url)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if attempt >= tile_retries:
                raise RuntimeError(
                    f"{provider} tile request failed for z/x/y={zoom}/{wrapped_x}/{y}: {body}"
                ) from exc
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            TimeoutError,
            OSError,
        ) as exc:
            if attempt >= tile_retries:
                raise RuntimeError(
                    f"{provider} tile request failed for z/x/y={zoom}/{wrapped_x}/{y}: {exc}"
                ) from exc
        time.sleep(min(2.0**attempt, 8.0))
    if data is None:
        raise RuntimeError(f"{provider} tile request failed for z/x/y={zoom}/{wrapped_x}/{y}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    if request_sleep_s > 0:
        time.sleep(request_sleep_s)
    return Image.open(BytesIO(data)).convert("RGB")


def build_preview(
    args: argparse.Namespace,
    api_key: str | None,
    session_token: str | None,
    area: dict,
    tile_size_px: int,
    out_dir: Path,
) -> tuple[Path, dict]:
    from PIL import Image

    preview_path = out_dir / f"{area['label']}_{args.provider}_satellite_50km_10000px.png"
    crop_box, native_mpp = source_crop_box(
        center_lat=area["center_lat"],
        center_lon=area["center_lon"],
        area_size_m=args.area_size_m,
        zoom=args.zoom,
        tile_size_px=tile_size_px,
    )
    x_range, y_range = tile_range_for_box(crop_box, tile_size_px)
    source_tile_count = len(x_range) * len(y_range)
    if source_tile_count > args.max_source_tiles:
        raise RuntimeError(
            f"{area['label']} would request {source_tile_count} source tiles. "
            f"Raise --max_source_tiles or lower --zoom."
        )

    metadata = {
        "label": area["label"],
        "description": area["description"],
        "center_lat": area["center_lat"],
        "center_lon": area["center_lon"],
        "area_size_m": args.area_size_m,
        "preview_size_px": args.preview_size_px,
        "preview_meters_per_pixel": args.area_size_m / args.preview_size_px,
        "provider": args.provider,
        "zoom": args.zoom,
        "source_tile_size_px": tile_size_px,
        "source_crop_box": list(crop_box),
        "native_meters_per_pixel_at_center": native_mpp,
        "source_tile_count": source_tile_count,
        "source_tile_x_range": [x_range.start, x_range.stop - 1],
        "source_tile_y_range": [y_range.start, y_range.stop - 1],
        "tile_url_template": args.tile_url_template if args.provider == "esri" else None,
        "tile_api": "Google Map Tiles API satellite 2D tiles"
        if args.provider == "google"
        else "Esri ArcGIS World Imagery XYZ tiles",
    }

    if args.skip_existing_preview and preview_path.exists():
        return preview_path, metadata

    mosaic_width = len(x_range) * tile_size_px
    mosaic_height = len(y_range) * tile_size_px
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))
    cache_dir = Path(args.tile_cache_dir) if args.tile_cache_dir else None

    tile_coords = [(tx, ty) for ty in y_range for tx in x_range]

    def fetch(coord: tuple[int, int]) -> tuple[int, int, Image.Image]:
        tx, ty = coord
        tile = download_tile(
                provider=args.provider,
                api_key=api_key,
                session_token=session_token,
                zoom=args.zoom,
                x=tx,
                y=ty,
                image_format=args.image_format,
                tile_url_template=args.tile_url_template,
                cache_dir=cache_dir,
                request_sleep_s=args.request_sleep_s,
                tile_retries=args.tile_retries,
            )
        return tx, ty, tile

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.tile_workers) as executor:
        futures = [executor.submit(fetch, coord) for coord in tile_coords]
        for future in concurrent.futures.as_completed(futures):
            tx, ty, tile = future.result()
            paste_x = (tx - x_range.start) * tile_size_px
            paste_y = (ty - y_range.start) * tile_size_px
            mosaic.paste(tile.resize((tile_size_px, tile_size_px)), (paste_x, paste_y))
            done += 1
            if done == 1 or done % 100 == 0 or done == source_tile_count:
                print(f"{area['label']}: downloaded {done}/{source_tile_count} source tiles")

    left, top, right, bottom = crop_box
    crop = mosaic.crop(
        (
            left - x_range.start * tile_size_px,
            top - y_range.start * tile_size_px,
            right - x_range.start * tile_size_px,
            bottom - y_range.start * tile_size_px,
        )
    )
    if crop.size != (args.preview_size_px, args.preview_size_px):
        crop = crop.resize((args.preview_size_px, args.preview_size_px), Image.Resampling.LANCZOS)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(preview_path)
    return preview_path, metadata


def compression_value(raw: str) -> str | None:
    return None if raw == "none" else raw


def append_covering_centers(size_px: int, tile_size_px: int, stride_px: int) -> list[int]:
    if tile_size_px > size_px:
        raise ValueError("--sthn_tile_size_px cannot exceed --preview_size_px")
    half = tile_size_px // 2
    centers = list(range(half, size_px - half + 1, stride_px))
    last = size_px - half
    if centers[-1] != last:
        centers.append(last)
    return centers


def write_sthn_h5(
    args: argparse.Namespace,
    preview_path: Path,
    area: dict,
    metadata: dict,
    out_dir: Path,
) -> Path:
    import h5py
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(preview_path).convert("RGB")
    if image.size != (args.preview_size_px, args.preview_size_px):
        raise ValueError(f"Preview has size {image.size}; expected square {args.preview_size_px}.")

    centers_y = append_covering_centers(
        args.preview_size_px, args.sthn_tile_size_px, args.sthn_tile_stride_px
    )
    centers_x = append_covering_centers(
        args.preview_size_px, args.sthn_tile_size_px, args.sthn_tile_stride_px
    )
    records = []
    preview_mpp = args.area_size_m / args.preview_size_px
    for row, center_y in enumerate(centers_y):
        for col, center_x in enumerate(centers_x):
            y_m = center_y * preview_mpp
            x_m = center_x * preview_mpp
            records.append(
                {
                    "name": (
                        f"@{y_m:.3f}@{x_m:.3f}@google_satellite@{area['label']}"
                        f"@r{row:02d}@c{col:02d}.png"
                    ),
                    "center_y": center_y,
                    "center_x": center_x,
                    "row": row,
                    "col": col,
                }
            )

    dataset_dir = out_dir / "sthn_tiles"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    h5_path = dataset_dir / f"{args.split}_database.h5"
    if h5_path.exists():
        h5_path.unlink()

    compression = compression_value(args.compression)
    names = [record["name"] for record in records]
    image_sizes = np.asarray(
        [[args.sthn_tile_size_px, args.sthn_tile_size_px]] * len(records), dtype=np.int32
    )

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["source"] = f"{args.provider}_satellite_tiles"
        h5.attrs["area_label"] = area["label"]
        h5.attrs["center_lat"] = area["center_lat"]
        h5.attrs["center_lon"] = area["center_lon"]
        h5.attrs["area_size_m"] = args.area_size_m
        h5.attrs["preview_size_px"] = args.preview_size_px
        h5.attrs["meters_per_pixel"] = preview_mpp
        h5.attrs["tile_size_px"] = args.sthn_tile_size_px
        h5.attrs["tile_stride_px"] = args.sthn_tile_stride_px
        h5.attrs["tile_size_m"] = args.sthn_tile_size_px * preview_mpp
        h5.attrs["tile_provider"] = args.provider
        h5.attrs["tile_zoom"] = args.zoom
        h5.create_dataset("image_name", data=names, dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("image_size", data=image_sizes, compression=compression)
        data = h5.create_dataset(
            "image_data",
            shape=(len(records), args.sthn_tile_size_px, args.sthn_tile_size_px, 3),
            dtype=np.uint8,
            chunks=(1, args.sthn_tile_size_px, args.sthn_tile_size_px, 3),
            compression=compression,
        )

        tile_dir = dataset_dir / "png_tiles"
        if args.export_sthn_tiles:
            tile_dir.mkdir(parents=True, exist_ok=True)

        half = args.sthn_tile_size_px // 2
        for index, record in enumerate(records):
            box = (
                record["center_x"] - half,
                record["center_y"] - half,
                record["center_x"] + half,
                record["center_y"] + half,
            )
            tile = image.crop(box).convert("RGB")
            data[index] = np.asarray(tile, dtype=np.uint8)
            if args.export_sthn_tiles:
                tile.save(tile_dir / f"r{record['row']:02d}_c{record['col']:02d}.png")

    metadata["sthn_h5_path"] = str(h5_path)
    metadata["sthn_tile_size_px"] = args.sthn_tile_size_px
    metadata["sthn_tile_stride_px"] = args.sthn_tile_stride_px
    metadata["sthn_tile_size_m"] = args.sthn_tile_size_px * preview_mpp
    metadata["sthn_tile_count"] = len(records)
    metadata["sthn_grid_rows"] = len(centers_y)
    metadata["sthn_grid_cols"] = len(centers_x)
    return h5_path


def write_manifest(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.area_size_m <= 0:
        raise ValueError("--area_size_m must be positive")
    if args.preview_size_px <= 0:
        raise ValueError("--preview_size_px must be positive")
    if args.sthn_tile_size_px <= 0 or args.sthn_tile_stride_px <= 0:
        raise ValueError("STHN tile size and stride must be positive")

    api_key = None
    session_token = None
    if args.provider == "google":
        api_key = require_api_key(args)
        session = create_google_session(args, api_key)
        session_token = session["session"]
        tile_size_px = int(session.get("tileWidth") or session.get("tileSize") or 256)
        print(f"Created Google satellite tile session; tile size is {tile_size_px}px")
    else:
        tile_size_px = 256
        print(f"Using Esri World Imagery tile endpoint; tile size is {tile_size_px}px")

    output_root = Path(args.output_root)
    for area in build_area_list(args):
        out_dir = output_root / area["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Building {area['label']} at {area['center_lat']}, {area['center_lon']}")
        preview_path, metadata = build_preview(
            args=args,
            api_key=api_key,
            session_token=session_token,
            area=area,
            tile_size_px=tile_size_px,
            out_dir=out_dir,
        )
        metadata["preview_path"] = str(preview_path)
        if args.write_h5:
            write_sthn_h5(args, preview_path, area, metadata, out_dir)
        manifest_path = out_dir / "manifest.json"
        write_manifest(manifest_path, metadata)
        print(f"Wrote preview: {preview_path}")
        print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

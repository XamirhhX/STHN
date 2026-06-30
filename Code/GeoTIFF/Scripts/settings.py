from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_SPLIT_RATIOS, DEFAULT_TILE_URL_TEMPLATE, DEFAULT_USER_AGENT
from .geo import area_bounds_around_center, validate_area_size, validate_center_lat_lon
from .output_paths import unique_output_folder
from .sthn_export import DEFAULT_STHN_DATASET_NAME
from .tiles import required_tiles

LOGGER = logging.getLogger(__name__)
STREET_TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


class SettingsError(ValueError):
    """Raised when setting.json contains invalid or incomplete values."""


@dataclass(frozen=True)
class RunSettings:
    mode: str
    center_lat: float
    center_lon: float
    area_size_meters: float
    zoom: int
    output_folder: Path
    unique_output_folder: bool
    output_size_px: int | None
    tile_url_template: str
    retries: int
    timeout_seconds: float
    user_agent: str
    max_tiles: int
    use_cache: bool
    patches_enabled: bool
    patch_size: int
    stride: int | None
    split_ratios: tuple[float, float, float]
    seed: int
    hdf5_enabled: bool
    hdf5_compression: str | None
    sthn_enabled: bool
    sthn_dataset_name: str
    sthn_database_size: int
    sthn_query_size: int
    sthn_stride: int | None
    sthn_compression: str | None

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_folder"] = str(self.output_folder)
        payload["split_ratios"] = list(self.split_ratios)
        return payload


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SettingsError(f"Settings file was not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SettingsError("Settings file must contain a JSON object.")
    return payload


def section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SettingsError(f"{name} must be a JSON object.")
    return value


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def as_str(value: Any, name: str) -> str:
    if value is None:
        raise SettingsError(f"{name} is required.")
    return str(value)


def as_float(value: Any, name: str) -> float:
    if value is None:
        raise SettingsError(f"{name} is required.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{name} must be a number.") from exc


def as_int(value: Any, name: str) -> int:
    if value is None:
        raise SettingsError(f"{name} is required.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{name} must be a whole number.") from exc


def as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return as_int(value, name)


def as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise SettingsError(f"{name} must be true or false.")


def as_path(value: Any, base_dir: Path, name: str) -> Path:
    raw = as_str(value, name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_split_ratios(value: Any) -> tuple[float, float, float]:
    from .patches import parse_split_ratios as parse

    return parse(value if value is not None else DEFAULT_SPLIT_RATIOS)


def normalize_compression(value: Any) -> str | None:
    if value is None:
        return "gzip"
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "false", "off"}:
        return None
    if normalized not in {"gzip", "lzf"}:
        raise SettingsError("hdf5.compression must be gzip, lzf, or none.")
    return normalized


def tile_url_template_from_settings(payload: dict[str, Any], imagery: dict[str, Any]) -> str:
    explicit_template = first_value(imagery.get("tile_url_template"), payload.get("tile_url_template"))
    if explicit_template is not None:
        return as_str(explicit_template, "imagery.tile_url_template")

    source = as_str(imagery.get("source", "satellite"), "imagery.source").strip().lower()
    if source == "satellite":
        return DEFAULT_TILE_URL_TEMPLATE
    if source == "street":
        return STREET_TILE_URL_TEMPLATE
    if source == "custom":
        raise SettingsError("imagery.source custom requires imagery.tile_url_template.")
    raise SettingsError("imagery.source must be satellite, street, or custom.")


def load_run_settings(path: Path) -> RunSettings:
    payload = load_json(path)
    base_dir = path.resolve().parent
    location = section(payload, "location")
    area = section(payload, "area")
    imagery = section(payload, "imagery")
    output = section(payload, "output")
    download = section(payload, "download")
    patches = section(payload, "patches")
    hdf5 = section(payload, "hdf5")
    sthn = section(payload, "sthn")

    mode = as_str(payload.get("mode", "geotiff"), "mode").strip().lower()
    if mode not in {"geotiff", "build", "full"}:
        raise SettingsError("mode must be geotiff, build, or full.")

    center_lat = as_float(
        first_value(
            location.get("latitude"),
            location.get("lat"),
            location.get("center_lat"),
            payload.get("latitude"),
            payload.get("center_lat"),
        ),
        "location.latitude",
    )
    center_lon = as_float(
        first_value(
            location.get("longitude"),
            location.get("lon"),
            location.get("center_lon"),
            payload.get("longitude"),
            payload.get("center_lon"),
        ),
        "location.longitude",
    )

    width_meters = first_value(area.get("width_meters"), area.get("size_meters"), payload.get("area_size_meters"))
    height_meters = first_value(area.get("height_meters"), area.get("size_meters"), payload.get("area_size_meters"))
    if width_meters is not None and height_meters is not None:
        width = as_float(width_meters, "area.width_meters")
        height = as_float(height_meters, "area.height_meters")
        if abs(width - height) > 1e-6:
            raise SettingsError("Only square GeoTIFF areas are supported. Use the same width_meters and height_meters.")
        area_size_meters = width
    else:
        area_size_meters = as_float(5000.0, "area.size_meters")

    output_size_px = as_optional_int(
        first_value(output.get("size_px"), output.get("output_size_px"), payload.get("output_size_px")),
        "output.size_px",
    )

    settings = RunSettings(
        mode=mode,
        center_lat=center_lat,
        center_lon=center_lon,
        area_size_meters=area_size_meters,
        zoom=as_int(first_value(imagery.get("zoom"), payload.get("zoom"), 16), "imagery.zoom"),
        output_folder=as_path(first_value(output.get("folder"), payload.get("output_folder"), "Results/from-settings"), base_dir, "output.folder"),
        unique_output_folder=as_bool(first_value(output.get("unique_folder"), output.get("unique"), True), "output.unique_folder"),
        output_size_px=output_size_px,
        tile_url_template=tile_url_template_from_settings(payload, imagery),
        retries=as_int(first_value(download.get("retries"), payload.get("retries"), 3), "download.retries"),
        timeout_seconds=as_float(
            first_value(download.get("timeout_seconds"), payload.get("timeout_seconds"), 20.0),
            "download.timeout_seconds",
        ),
        user_agent=as_str(first_value(download.get("user_agent"), payload.get("user_agent"), DEFAULT_USER_AGENT), "download.user_agent"),
        max_tiles=as_int(first_value(download.get("max_tiles"), payload.get("max_tiles"), 256), "download.max_tiles"),
        use_cache=as_bool(first_value(download.get("use_cache"), payload.get("use_cache"), True), "download.use_cache"),
        patches_enabled=as_bool(first_value(patches.get("enabled"), mode in {"build", "full"}), "patches.enabled"),
        patch_size=as_int(first_value(patches.get("patch_size"), 512), "patches.patch_size"),
        stride=as_optional_int(patches.get("stride"), "patches.stride"),
        split_ratios=parse_split_ratios(patches.get("split_ratios")),
        seed=as_int(first_value(patches.get("seed"), 1337), "patches.seed"),
        hdf5_enabled=as_bool(first_value(hdf5.get("enabled"), mode == "full"), "hdf5.enabled"),
        hdf5_compression=normalize_compression(hdf5.get("compression")),
        sthn_enabled=as_bool(first_value(sthn.get("enabled"), mode == "full"), "sthn.enabled"),
        sthn_dataset_name=as_str(
            first_value(sthn.get("dataset_name"), DEFAULT_STHN_DATASET_NAME),
            "sthn.dataset_name",
        ),
        sthn_database_size=as_int(first_value(sthn.get("database_size"), 1536), "sthn.database_size"),
        sthn_query_size=as_int(first_value(sthn.get("query_size"), 512), "sthn.query_size"),
        sthn_stride=as_optional_int(sthn.get("stride"), "sthn.stride"),
        sthn_compression=normalize_compression(sthn.get("compression")),
    )
    validate_run_settings(settings)
    return settings


def validate_run_settings(settings: RunSettings) -> None:
    validate_center_lat_lon(settings.center_lat, settings.center_lon)
    validate_area_size(settings.area_size_meters)
    if settings.zoom < 0:
        raise SettingsError("imagery.zoom must be non-negative.")
    if settings.output_size_px is not None and settings.output_size_px <= 0:
        raise SettingsError("output.size_px must be positive or null.")
    if settings.retries < 0:
        raise SettingsError("download.retries must be non-negative.")
    if settings.timeout_seconds <= 0:
        raise SettingsError("download.timeout_seconds must be positive.")
    if settings.max_tiles <= 0:
        raise SettingsError("download.max_tiles must be positive.")
    if settings.patch_size <= 0:
        raise SettingsError("patches.patch_size must be positive.")
    if settings.stride is not None and settings.stride <= 0:
        raise SettingsError("patches.stride must be positive or null.")
    if settings.hdf5_enabled and not settings.patches_enabled:
        raise SettingsError("hdf5.enabled requires patches.enabled to be true.")
    if settings.sthn_database_size <= 0:
        raise SettingsError("sthn.database_size must be positive.")
    if settings.sthn_query_size <= 0:
        raise SettingsError("sthn.query_size must be positive.")
    if settings.sthn_stride is not None and settings.sthn_stride <= 0:
        raise SettingsError("sthn.stride must be positive or null.")


def estimate_tile_count(settings: RunSettings) -> int:
    bounds = area_bounds_around_center(settings.center_lat, settings.center_lon, settings.area_size_meters)
    return len(required_tiles(bounds, settings.zoom).tiles)


def run_settings(settings: RunSettings) -> Path:
    from .geotiff import build_geotiff
    from .hdf5_export import export_h5
    from .patches import generate_patches
    from .sthn_export import export_sthn_dataset

    output_folder = unique_output_folder(settings.output_folder) if settings.unique_output_folder else settings.output_folder
    if output_folder != settings.output_folder:
        LOGGER.info("Output folder exists; writing this run to: %s", output_folder)

    geotiff_result = build_geotiff(
        center_lat=settings.center_lat,
        center_lon=settings.center_lon,
        area_size_meters=settings.area_size_meters,
        zoom=settings.zoom,
        output_folder=output_folder,
        tile_url_template=settings.tile_url_template,
        retries=settings.retries,
        timeout_seconds=settings.timeout_seconds,
        user_agent=settings.user_agent,
        max_tiles=settings.max_tiles,
        use_cache=settings.use_cache,
        output_size_px=settings.output_size_px,
    )

    if settings.patches_enabled:
        dataset_folder = output_folder / "Patch_Dataset"
        generate_patches(
            geotiff_path=geotiff_result.geotiff_path,
            output_folder=dataset_folder,
            patch_size=settings.patch_size,
            stride=settings.stride,
            split_ratios=settings.split_ratios,
            seed=settings.seed,
        )
        if settings.hdf5_enabled:
            export_h5(
                dataset_folder=dataset_folder,
                output_folder=dataset_folder,
                compression=settings.hdf5_compression,
            )

    if settings.sthn_enabled:
        export_sthn_dataset(
            geotiff_path=geotiff_result.geotiff_path,
            output_folder=output_folder / "STHN_Model_Input",
            dataset_name=settings.sthn_dataset_name,
            database_size=settings.sthn_database_size,
            query_size=settings.sthn_query_size,
            stride=settings.sthn_stride,
            split_ratios=settings.split_ratios,
            seed=settings.seed,
            compression=settings.sthn_compression,
        )
    LOGGER.info("Settings run finished: %s", output_folder)
    return geotiff_result.geotiff_path

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from pyproj import Transformer

WEB_MERCATOR_RADIUS_M = 6378137.0
WEB_MERCATOR_ORIGIN_SHIFT_M = pi * WEB_MERCATOR_RADIUS_M
WEB_MERCATOR_MAX_LAT = 85.05112878
TILE_SIZE_PX = 256
MAX_AREA_SIZE_METERS = 5_000.0

WGS84_TO_WEB_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
WEB_MERCATOR_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class Bounds3857:
    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def to_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "top": self.top,
        }


@dataclass(frozen=True)
class Bounds4326:
    west: float
    south: float
    east: float
    north: float

    def to_dict(self) -> dict[str, float]:
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }


def validate_center_lat_lon(center_lat: float, center_lon: float) -> None:
    if not -WEB_MERCATOR_MAX_LAT <= center_lat <= WEB_MERCATOR_MAX_LAT:
        msg = f"center_lat must be within Web Mercator limits +/-{WEB_MERCATOR_MAX_LAT}."
        raise ValueError(msg)
    if not -180.0 <= center_lon <= 180.0:
        raise ValueError("center_lon must be in [-180, 180].")


def validate_area_size(area_size_meters: float) -> None:
    if area_size_meters <= 0:
        raise ValueError("area_size_meters must be positive.")
    if area_size_meters > MAX_AREA_SIZE_METERS:
        msg = f"area_size_meters must be <= {MAX_AREA_SIZE_METERS:g}."
        raise ValueError(msg)


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    x, y = WGS84_TO_WEB_MERCATOR.transform(lon, lat)
    return float(x), float(y)


def web_mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon, lat = WEB_MERCATOR_TO_WGS84.transform(x, y)
    return float(lon), float(lat)


def area_bounds_around_center(
    center_lat: float,
    center_lon: float,
    area_size_meters: float,
) -> Bounds3857:
    validate_center_lat_lon(center_lat, center_lon)
    validate_area_size(area_size_meters)
    center_x, center_y = lonlat_to_web_mercator(center_lon, center_lat)
    half_size = area_size_meters / 2.0
    bounds = Bounds3857(
        left=center_x - half_size,
        bottom=center_y - half_size,
        right=center_x + half_size,
        top=center_y + half_size,
    )
    if (
        bounds.left < -WEB_MERCATOR_ORIGIN_SHIFT_M
        or bounds.right > WEB_MERCATOR_ORIGIN_SHIFT_M
        or bounds.bottom < -WEB_MERCATOR_ORIGIN_SHIFT_M
        or bounds.top > WEB_MERCATOR_ORIGIN_SHIFT_M
    ):
        raise ValueError("Requested area extends outside the valid EPSG:3857 extent.")
    return bounds


def bounds_3857_to_4326(bounds: Bounds3857) -> Bounds4326:
    west, south = web_mercator_to_lonlat(bounds.left, bounds.bottom)
    east, north = web_mercator_to_lonlat(bounds.right, bounds.top)
    return Bounds4326(west=west, south=south, east=east, north=north)


def mercator_resolution_m_per_pixel(zoom: int) -> float:
    if zoom < 0:
        raise ValueError("zoom must be non-negative.")
    world_px = TILE_SIZE_PX * (2**zoom)
    return (2.0 * WEB_MERCATOR_ORIGIN_SHIFT_M) / float(world_px)


def web_mercator_to_global_pixel(x: float, y: float, zoom: int) -> tuple[float, float]:
    resolution = mercator_resolution_m_per_pixel(zoom)
    px = (x + WEB_MERCATOR_ORIGIN_SHIFT_M) / resolution
    py = (WEB_MERCATOR_ORIGIN_SHIFT_M - y) / resolution
    return px, py


def global_pixel_to_web_mercator(px: float, py: float, zoom: int) -> tuple[float, float]:
    resolution = mercator_resolution_m_per_pixel(zoom)
    x = (px * resolution) - WEB_MERCATOR_ORIGIN_SHIFT_M
    y = WEB_MERCATOR_ORIGIN_SHIFT_M - (py * resolution)
    return x, y

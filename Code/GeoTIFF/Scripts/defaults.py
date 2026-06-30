from __future__ import annotations

DEFAULT_TILE_URL_TEMPLATE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
DEFAULT_USER_AGENT = "geotiff-scripts/0.1 (+https://example.invalid/local-tool)"
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "val", "test")

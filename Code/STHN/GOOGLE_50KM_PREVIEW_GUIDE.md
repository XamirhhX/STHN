# Google 50 km Preview And STHN Tile Guide

This workflow replaces the previous Bing/source-map assumption for the large
preview image. It can use either the official Google Map Tiles API satellite
layer or the same no-key Esri World Imagery style endpoint used by the local
GeoTIFF project.

## What It Builds

For each requested area, the script creates:

```text
datasets/google_50km_previews/
  lut_desert_hard/
    lut_desert_hard_google_satellite_50km_10000px.png
    manifest.json
    sthn_tiles/
      test_database.h5
  tehran_city/
    tehran_city_google_satellite_50km_10000px.png
    manifest.json
    sthn_tiles/
      test_database.h5
```

The preview PNG is exactly `10000 x 10000` pixels and represents `50 x 50 km`,
so the preview resolution is `5 m/px`.

The STHN tile H5 uses `1536 x 1536` RGB satellite chips by default. That matches
the larger STHN database input size used by the two-stage local scripts.

## Requirements

Set a Google Maps Platform API key with Map Tiles API enabled:

```powershell
$env:GOOGLE_MAPS_API_KEY="YOUR_KEY_HERE"
```

For Google, the script intentionally uses the official Map Tiles API session
endpoint rather than undocumented tile URLs.

For no-key testing, use:

```powershell
python scripts/build_google_50km_previews.py `
  --provider esri `
  --areas desert city `
  --output_root datasets/esri_50km_previews `
  --area_size_m 50000 `
  --preview_size_px 10000 `
  --zoom 15 `
  --sthn_tile_size_px 1536 `
  --sthn_tile_stride_px 1536
```

## Build Desert And City Previews

From the repository root:

```powershell
python scripts/build_google_50km_previews.py `
  --provider google `
  --areas desert city `
  --output_root datasets/google_50km_previews `
  --area_size_m 50000 `
  --preview_size_px 10000 `
  --zoom 15 `
  --sthn_tile_size_px 1536 `
  --sthn_tile_stride_px 1536
```

Default centers:

| Area | Label | Center |
|---|---|---|
| Hard desert | `lut_desert_hard` | `30.886021, 57.895202` |
| City | `tehran_city` | `35.689200, 51.389000` |

## Custom Centers

Add another 50 km area with:

```powershell
python scripts/build_google_50km_previews.py `
  --custom_area my_area,31.0,58.0
```

## Outputs To Use Next

Use the preview PNG when you need the large visual map:

```text
datasets/google_50km_previews/lut_desert_hard/lut_desert_hard_google_satellite_50km_10000px.png
```

Use the H5 file when you need STHN-sized candidate satellite chips:

```text
datasets/google_50km_previews/lut_desert_hard/sthn_tiles/test_database.h5
```

The H5 contains:

- `image_data`: RGB satellite chips, `1536 x 1536 x 3`
- `image_name`: metric center coordinates encoded as `@y_m@x_m@...`
- `image_size`: chip dimensions
- attrs for source, center, preview resolution, tile stride, and Google zoom

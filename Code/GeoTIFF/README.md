# GeoTIFF Scripts

Lightweight CLI tools for creating small satellite/map-tile GeoTIFFs, organized result folders, and STHN-compatible model input datasets for homography-based geolocation experiments.

The project intentionally has no GUI and no heavy framework. It is designed for small areas only: the CLI rejects areas larger than 5 km x 5 km and limits tile downloads by default.

## Install

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run From `setting.json`

Edit `setting.json`, especially:

- `location.latitude`
- `location.longitude`
- `area.width_meters` and `area.height_meters`
- `imagery.zoom`
- `output.folder`
- `output.size_px`

The included default creates a satellite GeoTIFF for a 5 km x 5 km square centered on the configured latitude/longitude.

Preview the resolved configuration without downloading tiles:

```powershell
.\.venv\Scripts\python.exe main.py from-settings --dry-run
```

Run it:

```powershell
.\.venv\Scripts\python.exe main.py from-settings
```

If the configured output folder already contains files, `from-settings` creates a timestamped sibling folder instead of overwriting it. Use `--reuse-output-folder` only when you intentionally want to write into the existing folder.

Or use the Windows helper:

```powershell
.\Run.bat
```

You can also override the center point without editing the file:

```powershell
.\Run.bat --center-lat 35.6892 --center-lon 51.3890
```

## Quick Start

Build a GeoTIFF around a center point:

```powershell
python main.py geotiff --center-lat 35.6892 --center-lon 51.3890 --output-folder Results\tehran
```

Run the full pipeline: GeoTIFF, PNG patches, generic HDF5 split files, and STHN model input:

```powershell
python main.py build --center-lat 35.6892 --center-lon 51.3890 --output-folder Results\tehran --export-sthn
```

Generate patches from an existing GeoTIFF:

```powershell
python main.py patches --geotiff Results\tehran\GeoTIFF\satellite.tif --output-folder Results\tehran\Patch_Dataset
```

Export existing patch manifests to HDF5:

```powershell
python main.py export-h5 --dataset-folder Results\tehran\Patch_Dataset
```

Export an existing GeoTIFF directly into the STHN loader layout:

```powershell
python main.py export-sthn --geotiff Results\tehran\GeoTIFF\satellite.tif --output-folder Results\tehran\STHN_Model_Input
```

New runs use this readable layout:

```text
Results/<run-name>/
  GeoTIFF/satellite.tif
  Preview/satellite_preview.png
  Metadata/geotiff_metadata.json
  Tiles/
  Patch_Dataset/
  STHN_Model_Input/
```

## CLI Commands

### `from-settings`

Runs the project from `setting.json`. This is the easiest way to get a satellite GeoTIFF for a configured center latitude/longitude and 5 km x 5 km area.

```powershell
python main.py from-settings
```

Useful options:

```text
--settings PATH             default: setting.json
--dry-run                   print resolved settings and estimated tile count
--center-lat FLOAT          override setting.json latitude
--center-lon FLOAT          override setting.json longitude
--area-size-meters FLOAT    override square area size, maximum: 5000
--zoom INT                  override imagery zoom
--output-size-px INT        override square output pixels
--output-folder PATH        override output folder
--reuse-output-folder       allow writing into an existing output folder
```

### `geotiff`

Downloads the minimum required OSM-compatible tiles, stitches them, crops to the requested area, writes:

- `GeoTIFF/satellite.tif`
- `Preview/satellite_preview.png`
- `Metadata/geotiff_metadata.json`

Important options:

```text
--center-lat FLOAT           required
--center-lon FLOAT           required
--area-size-meters FLOAT     default: 500, maximum: 5000
--zoom INT                   default: 18
--output-size-px INT         optional, e.g. 512 for exact STHN chips
--output-folder PATH         default: Results/latest
--tile-url-template TEXT     default: https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
--max-tiles INT              default: 256
```

The tile URL template supports `{z}`, `{x}`, `{y}`, and `{quadkey}`. The default is a Web Mercator satellite imagery layer. You can also use OSM-compatible services or Bing-style quadkey templates when you provide the appropriate URL and key.

`--area-size-meters` controls the Earth coverage. `--output-size-px` controls the final square image dimensions. `--zoom` controls source tile detail before resampling.

### `build`

Runs `geotiff`, then `patches`, then `export-h5`. Add `--export-sthn` to also write the model-input directory.

```powershell
python main.py build --center-lat 35.6892 --center-lon 51.3890 --output-folder Results\sample --patch-size 512 --export-sthn
```

Use `--skip-patches` or `--skip-h5` if you only want part of the pipeline.

### `patches`

Creates a deterministic patch dataset:

```text
Patch_Dataset/
  images/
  metadata/
  train/
  val/
  test/
```

Each patch has:

- PNG image crop
- JSON metadata
- center GPS coordinates
- EPSG:3857 and EPSG:4326 bounding boxes
- pixel resolution

Default patch size is `512`. Use `--patch-size 1024` for larger crops when the GeoTIFF is large enough.

### `export-h5`

Creates:

- `train.h5`
- `val.h5`
- `test.h5`

Each file contains:

- `images`: uint8 patch tensors
- `gps`: center latitude/longitude
- `bbox_epsg3857`: left, bottom, right, top
- `bbox_epsg4326`: west, south, east, north
- `ids`: patch identifiers

### `export-sthn`

Creates a directory that can be passed to the STHN repo loaders:

```text
STHN_Model_Input/
  sthn_dataset/
    maps/satellite/20201117_BingSatellite.png
    satellite_0_thermalmapping_135_train/
      train_database.h5
      train_queries.h5
      val_database.h5
      val_queries.h5
      test_database.h5
      test_queries.h5
```

The generated query images are synthetic grayscale chips derived from the satellite raster. This makes the output loadable by the model pipeline, but it is not a replacement for real thermal imagery.

## Notes

- Default CRS is EPSG:3857.
- The final GeoTIFF crop is snapped to the Web Mercator pixel grid at the requested zoom. This preserves a correct affine transform and pixel-to-meter alignment.
- For production or repeated downloads, use your own tile service or cache and respect the provider's tile usage policy.
- This is meant for small STHN experiments, not bulk map scraping.

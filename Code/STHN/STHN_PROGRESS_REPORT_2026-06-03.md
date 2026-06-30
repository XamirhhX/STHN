# STHN Progress Report

Date: 2026-06-03

Author: Reacher

## Scope

This report documents three work streams completed around the STHN project:

1. Running small Colab experiments on the scaled thermal observation setup.
2. Downloading and organizing a 1000-sample thermal/satellite image set for GeoTIFF preparation.
3. Studying the model entry-point architecture and how inference is assembled.

The work is focused on practical experimentation and implementation understanding, not on summarizing the STHN paper.

## 1. Colab Tiny Experiments On Scaled Thermal Observations

I ran a small smoke-test experiment using the scaled-observation H5 workflow. The local result folder is:

```text
scaled_observation_h5_2026-05-19_15-40-28/
```

Important generated artifacts:

```text
scaled_observation_h5_2026-05-19_15-40-28/
  scaled_observation_h5_results.csv
  SCALE_OBSERVATION_H5_SMOKE_TEST_ANALYSIS.md
  mace_vs_missing_ratio.png
  mace_vs_scale.png
  examples/
```

The experiment used the frozen pretrained two-stage STHN model:

```text
xjh19972/STHN/two_stages
```

The tested dataset folder was:

```text
satellite_0_thermalmapping_135
```

The experiment was intentionally small. It evaluated 3 variations, not the final intended 100-sample or 1000-sample evaluation. The purpose was to verify that the Colab/H5/model pipeline worked end to end before scaling up.

### Experiment Configuration

The scaled-observation test simulated a thermal observation that covers a smaller footprint than the normal model input. The thermal query was scaled to 80 percent linear size, corresponding to a simulated "20 percent closer" condition, and then placed into a 256 x 256 canvas as a two-tile diagonal mosaic.

Key settings:

| Setting | Value |
|---|---:|
| Number of evaluated variations | 3 |
| Scale | 0.8 |
| Simulated closer percent | 20 percent |
| Number of tiles | 2 |
| Tile path | diagonal |
| Fill modes | zero, half, mean |
| Mosaic canvas | 256 x 256 |
| Effective covered area | 92.06 percent |
| Effective missing/no-data area | 7.94 percent |

Tile placements:

```text
Tile 1: [0, 0, 205, 205]
Tile 2: [51, 51, 256, 256]
```

### Results

The main measured alignment errors were:

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| MACE, resized 256 px space | 23.62 px | 24.78 px | 13.99 px | 32.07 px |
| MACE, database scale | 141.69 px | 148.70 px | 83.94 px | 192.43 px |
| Center error, database scale | 122.82 px | 132.16 px | 53.10 px | 183.21 px |
| Normalized MACE vs 1536 px patch | 9.22 percent | 9.68 percent | 5.46 percent | 12.53 percent |
| MACE relative to ground-truth motion magnitude | 19.56 percent | 20.52 percent | 11.59 percent | 26.56 percent |

Per-sample results:

| Variation | Dataset index | Positive index | Fill mode | MACE database | Center error database | MACE / GT motion | Area ratio | Invalid geometry |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 0 | 212 | zero | 83.94 px | 53.10 px | 11.59 percent | 6.76 percent | false |
| 1 | 1 | 2 | half | 192.43 px | 183.21 px | 26.56 percent | 6.05 percent | false |
| 2 | 2 | 213 | mean | 148.70 px | 132.16 px | 20.52 percent | 6.12 percent | false |

### Interpretation

The smoke test proved the following:

- The H5 data path was readable in Colab.
- The `satellite_0_thermalmapping_135` dataset variant could be used for inference.
- The Hugging Face two-stage checkpoint loaded successfully.
- The scaled thermal mosaic generation code worked on real H5 thermal query images.
- The frozen STHN model produced predictions on the scaled/mosaicked thermal inputs.
- CSV metrics and diagnostic plots were generated.

The test also exposed a geometry concern. The `flag_invalid_geometry` field was false for all 3 samples, but the predicted quadrilateral area ratios were only about 6 percent. That is suspicious because the area is very small relative to the expected patch footprint. For larger experiments, predictions with very small area ratio or high MACE should be inspected manually.

The local result folder should be treated as a smoke-test artifact, not a final benchmark.

## 2. 1000 Thermal/Satellite Images For GeoTIFF Preparation

I worked on downloading and organizing a set of 1000 thermal and satellite samples from the database so they can be used to design a GeoTIFF construction workflow.

The relevant local supporting documents are:

```text
FULL_DATASET_DOWNLOAD_AND_SCALE_RUN.md
STHN_H5_DATABASE_GUIDE.md
```

The full dataset download path documented for Colab uses the Hugging Face dataset:

```text
xjh19972/boson-nighttime
satellite-thermal-dataset-v3/satellite_thermal_dataset_v3.tar.gz.part*
```

The guide notes that the full archive is large, roughly 130 GB, and should be reconstructed from multipart archive files before extraction.

### Dataset Structure Identified

The STHN local pipeline does not treat each satellite crop as an independent full-resolution GeoTIFF. Instead, the dataset has:

```text
maps/satellite/20201117_BingSatellite.png
satellite_0_thermalmapping_135_train/test_queries.h5
satellite_0_thermalmapping_135_train/test_database.h5
```

The H5 files provide image names and metadata:

- `test_queries.h5` contains thermal/query image data and query names.
- `test_database.h5` contains database/map crop names and database coordinates.
- `maps/satellite/20201117_BingSatellite.png` is the large satellite map from which satellite database crops are extracted.

The repository parses coordinate-like metadata from the H5 `image_name` values by splitting names on `@`. In the local data loader, the database H5 is mainly used for candidate names and coordinates. The actual satellite image crop is extracted from the large satellite map PNG.

This means the GeoTIFF structure should be built around the large satellite map and its coordinate transform, not around disconnected satellite image crops alone.

### Proposed 1000-Sample Organization

The 1000 downloaded records should be organized as paired query/database samples:

```text
geotiff_preparation_1000/
  satellite/
    source_map/
      20201117_BingSatellite.png
    crops/
      sample_000000_sat.png
      sample_000001_sat.png
      ...
  thermal/
    queries/
      sample_000000_thermal.png
      sample_000001_thermal.png
      ...
  metadata/
    pairs.csv
    pairs.jsonl
    crs.json
    affine_transform.json
    notes.md
```

Recommended `pairs.csv` fields:

```text
sample_id
split
query_h5_path
database_h5_path
query_image_name
database_image_name
thermal_image_path
satellite_crop_path
query_utm_easting
query_utm_northing
database_utm_easting
database_utm_northing
map_pixel_x
map_pixel_y
crop_left
crop_top
crop_right
crop_bottom
database_size
thermal_size
source_map_path
```

For compatibility with STHN inference, satellite crops should preserve the trained database crop convention:

```text
database_size = 1536
thermal_size = 256
```

The model expects the satellite crop as the larger search window and the thermal image as the 256 x 256 query. Any GeoTIFF preparation should preserve that scale relationship unless a controlled resampling strategy is used.

### GeoTIFF Construction Plan

The practical GeoTIFF plan is:

1. Use the large satellite map as the georeferenced raster source.
2. Determine or recover the CRS and affine transform for the satellite map.
3. Convert each database H5 coordinate into map pixel coordinates.
4. Extract `1536 x 1536` satellite crops centered on database coordinates.
5. Store the thermal query image separately as an observation image, not as a map raster unless independent camera pose/georeferencing is available.
6. Build a metadata table linking each thermal query to its candidate satellite crop, UTM-like coordinate tokens, pixel crop bounds, and source H5 row.
7. If a full GeoTIFF mosaic is needed, georeference the large satellite map or a stitched satellite mosaic first, then use STHN predictions to map thermal observation corners into that GeoTIFF coordinate space.

Important constraint: the thermal images are not automatically GeoTIFF map layers. They are observation/query images. They can be projected into GeoTIFF/world coordinates only after STHN predicts a homography and the predicted satellite crop corners are converted through the map affine transform.

### Current Status

The completed part is the data acquisition and structure planning work. The final GeoTIFF artifact still requires a verified CRS/affine transform and a validation step showing that crop centers round-trip correctly between:

```text
H5 coordinate token -> satellite map pixel -> crop bounds -> GeoTIFF/world coordinate
```

## 3. Studying Entry-Point Architecture Of The Model

I studied the STHN implementation architecture from the source code and produced the detailed architecture report:

```text
STHN_IMPLEMENTATION_ARCHITECTURE.md
```

The main files inspected for the entry-point and inference path were:

```text
STHN_demo.py
experiments/scaled_observation_eval.py
experiments/scaled_observation_utils.py
local_pipeline/model/network.py
local_pipeline/datasets_4cor_img.py
global_pipeline/test.py
STHN_H5_DATABASE_GUIDE.md
```

### Main Entry Points

The cleanest pretrained inference entry point is:

```text
STHN_demo.py
```

This loads the two-stage pretrained STHN model, preprocesses a satellite image and thermal image, runs inference, and visualizes the predicted alignment.

The scaled-observation experiment entry point is:

```text
experiments/scaled_observation_eval.py
```

This loads real H5 samples, builds scaled thermal mosaics, runs the frozen pretrained STHN model, and writes CSV/plot outputs.

The H5 data path is assembled through:

```text
local_pipeline/datasets_4cor_img.py
```

This file reads thermal query images from H5 and crops satellite database regions from the large satellite map.

### Architecture Findings

The model does not search the whole satellite map internally. It receives an already selected satellite crop and a thermal query image.

The core inference path is:

```text
satellite crop [B, 3, 1536, 1536]
thermal query [B, 3, 256, 256]
        |
resize satellite crop to 256 x 256
        |
shared feature encoder
        |
satellite features and thermal features
        |
correlation / matching block
        |
iterative update block
        |
four-corner displacement
        |
homography estimate
        |
optional two-stage satellite crop refinement
        |
final four-corner prediction
```

The first meaningful interaction between satellite and thermal images happens inside the correlation block after both images have been encoded by the shared CNN feature extractor.

The most important implementation conclusion is that localization is split across two levels:

- External candidate selection decides which satellite crop is passed into the model.
- STHN estimates the homography inside that selected crop.

Therefore, search biasing, geographic priors, candidate filtering, and GeoTIFF routing can be added without retraining if they are inserted before crop generation or after model prediction.

### Inference-Time Modification Points

Safe no-retraining modification points identified during the architecture study:

| Modification point | Safety | Reason |
|---|---|---|
| Candidate satellite crop generation | High | Occurs before neural inference |
| GeoTIFF/GIS coordinate conversion | High | Pure preprocessing/postprocessing |
| Candidate ranking and filtering | High | External to learned weights |
| Thermal frame stitching or mosaicking | Medium | Input distribution changes, but no checkpoint change required |
| Repeating or controlling refinement calls | Medium | Uses frozen modules but may alter assumptions |
| Feature caching across candidates | Medium | Requires careful preservation of tensor conventions |
| Internal encoder/correlation/update replacement | Low | Tightly coupled to learned weights |

The safest research direction is to build a map-aware crop router around STHN instead of changing STHN's trained internal modules.

## Overall Outcome

The work completed so far establishes a practical path for steering STHN without retraining:

1. The Colab smoke test confirmed that frozen STHN inference can run on real H5 samples and on scaled thermal mosaics.
2. The 1000-image dataset preparation clarified how thermal query images, satellite map crops, H5 coordinate metadata, and GeoTIFF structure should be connected.
3. The architecture study showed that STHN is best treated as a crop-level homography estimator, while global/geographic search should be handled externally.

## Next Recommended Steps

1. Run the scaled-observation experiment with at least 100 samples using `experiments/scaled_observation_eval.py`.
2. Export the 1000-sample metadata table with crop bounds and parsed coordinate fields.
3. Verify the satellite map CRS and affine transform needed for GeoTIFF generation.
4. Build a GeoTIFF-aware crop generator that converts GPS/UTM priors into candidate `1536 x 1536` satellite crops.
5. Run frozen STHN over multiple candidate crops per thermal query and rank candidates using predicted homography quality, area ratio, center error proxies, and geographic prior consistency.

## Referenced Local Artifacts

```text
scaled_observation_h5_2026-05-19_15-40-28/
scaled_observation_h5_2026-05-19_15-40-28/scaled_observation_h5_results.csv
scaled_observation_h5_2026-05-19_15-40-28/SCALE_OBSERVATION_H5_SMOKE_TEST_ANALYSIS.md
FULL_DATASET_DOWNLOAD_AND_SCALE_RUN.md
STHN_H5_DATABASE_GUIDE.md
STHN_IMPLEMENTATION_ARCHITECTURE.md
experiments/scaled_observation_eval.py
experiments/scaled_observation_utils.py
local_pipeline/datasets_4cor_img.py
local_pipeline/model/network.py
STHN_demo.py
```

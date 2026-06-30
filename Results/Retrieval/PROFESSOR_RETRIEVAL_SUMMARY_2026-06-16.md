# Professor Retrieval Summary - 2026-06-16

## What The Professor Wants

From the professor notes, discussion notes, and the three papers, the immediate task is the retrieval/localization stage before homography refinement.

Paper alignment:

- First paper: global satellite-thermal retrieval. This is the stage to reproduce first.
- Second paper: STHN deep homography refinement after candidate satellite crops are selected.
- Third paper: UASTHN uncertainty/failure detection after homography, especially for textureless, self-similar, corrupted, outdated, or out-of-area cases.

Operational interpretation:

- Input: a UAV/drone query image and an estimated prior location.
- Database: precomputed embeddings for Level-16/17 satellite tiles.
- Search: start at the prior and evaluate tiles outward in a spiral.
- Output: Top-1, Top-3, Top-5 candidates, runtime, and a Top-3 plus overlap-tile handoff list for STHN/IHN.
- Professor-scale target: two 25 x 25 grids, where the second grid is shifted by half a tile period, giving 625 + 625 = 1,250 candidate tiles.

## Implemented

Code added or updated:

- `C:\Users\Asus\Documents\Project\STHN-main\STHN-main\scripts\professor_retrieval_benchmark.py`
- `C:\Users\Asus\Documents\Project\STHN-main\STHN-main\scripts\create_professor_scale_h5_from_tile_cache.py`
- `C:\Users\Asus\Documents\Project\STHN-main\STHN-main\scripts\build_professor_scale_geotiff.py`

Main functionality:

- Builds a retrieval database from STHN/GeoTIFF H5 files or from cached Web Mercator tiles.
- Uses CUDA ResNet-18 ImageNet embeddings on the RTX 4060 Laptop GPU.
- Supports H5 coordinates in pixels or meters.
- Runs prior-centered spiral search.
- Reports Top-1, Top-3, Top-5 accuracy, precision, recall, Top-1 center error, runtime, FPS, and evaluated tile counts.
- Writes ranked retrieval CSV and homography handoff CSV.
- Expands Top-3 candidates with nearest overlap tiles from the other shifted grid.

The professor-scale GeoTIFF build previously failed because PIL blocked the huge 52 km image as a decompression-bomb risk. The launcher now sets `Image.MAX_IMAGE_PIXELS = None`. To avoid needing the huge intermediate GeoTIFF, I also generated the professor-scale H5 directly from the already downloaded Level-16 tile cache.

## Test Data

| Dataset | Terrain | Area | Database | Queries | Notes |
|---|---|---:|---:|---:|---|
| San Francisco | Urban | 5 km x 5 km | 98 | 36 | Existing compact two-grid GeoTIFF test bundle |
| Grand Canyon | Mountains | 5 km x 5 km | 196 | 196 | Existing GeoTIFF/STHN export |
| Lut Desert / Kaluts | Desert with features | 50 km x 50 km | 1,250 | 625 | Professor-scale two-grid Level-16 cache-derived H5 |

Professor-scale dataset created:

- `C:\Users\Asus\Documents\Project\Retrieval_Professor_Scale_Bundle_2026-06-15\datasets\retrieval_iran_lut_kaluts_50km_two_grid\test_database.h5`
- `C:\Users\Asus\Documents\Project\Retrieval_Professor_Scale_Bundle_2026-06-15\datasets\retrieval_iran_lut_kaluts_50km_two_grid\test_queries.h5`
- `C:\Users\Asus\Documents\Project\Retrieval_Professor_Scale_Bundle_2026-06-15\datasets\retrieval_iran_lut_kaluts_50km_two_grid\test_tile_cache_manifest.json`

The Kaluts database uses 2,000 m tile periods and two half-shifted grids. Queries are grayscale 1,024 m crops resized to 512 x 512 as a simple thermal-like proxy.

## Results

| Dataset | Top-1 | Top-3 | Top-5 | Mean ms/query | Median ms/query | FPS | Mean tiles evaluated |
|---|---:|---:|---:|---:|---:|---:|---:|
| San Francisco urban | 97.22% | 100.00% | 100.00% | 3.51 | 1.05 | 285.07 | 97.97 |
| Grand Canyon mountains | 57.14% | 68.88% | 77.55% | 2.51 | 2.09 | 399.19 | 195.29 |
| Kaluts 50 km professor-scale | 18.72% | 38.40% | 44.64% | 4.51 | 4.10 | 221.64 | 1,219.70 |

Result files:

- `C:\Users\Asus\Documents\Project\04_Results\Retrieval\san_francisco_urban_allqueries_warm_20260616\professor_retrieval_report.json`
- `C:\Users\Asus\Documents\Project\04_Results\Retrieval\grand_canyon_mountains_allqueries_warm_20260616\professor_retrieval_report.json`
- `C:\Users\Asus\Documents\Project\04_Results\Retrieval\iran_lut_kaluts_professor_scale_warm_20260616\professor_retrieval_report.json`

Each result folder also contains:

- `professor_retrieval_rankings.csv`
- `professor_homography_candidates.csv`
- `PROFESSOR_RETRIEVAL_BENCHMARK.md`

## Interpretation

Runtime is better than the professor's rough 100 FPS assumption. The full 1,250-tile Kaluts search ran at about 222 FPS mean and 4.10 ms median per query on the RTX 4060 Laptop GPU.

Accuracy depends strongly on terrain:

- Urban San Francisco is easy for this baseline, with Top-3 and Top-5 at 100%.
- Grand Canyon is much harder because the terrain is repetitive and visually similar.
- Kaluts is the most professor-relevant stress test. It is desert with features, but many tiles are self-similar, and the query is only a grayscale optical proxy rather than real thermal or a trained satellite-thermal embedding. The low Top-1/Top-5 result is expected for a plain ImageNet ResNet baseline.

Important caveat: this is a retrieval-system benchmark and baseline descriptor implementation, not a trained reproduction of the 2023 satellite-thermal retrieval model. No trained global satellite-thermal checkpoint was found locally. The pipeline is now in place, and replacing the descriptor with the paper's trained encoder or the STHN global retrieval checkpoint should improve the difficult-terrain results.

## Accuracy Definition

A retrieval is counted correct when the query center lies inside the retrieved tile footprint. For shifted two-grid data, multiple database tiles can be valid positives because overlapping shifted-grid tiles can cover the same query center. Therefore:

- Top-K accuracy answers the professor's main question: "Does a correct tile appear in Top-K?"
- Precision and recall are also written, but recall can look lower when a query has several valid overlapping positives.

## Next Step

The highest-value next step is to replace the ImageNet ResNet baseline with the trained global satellite-thermal retrieval encoder from the first paper/STHN global pipeline, then rerun the same three result folders. The pipeline, data, metrics, spiral search, and homography handoff files are already ready for that swap.

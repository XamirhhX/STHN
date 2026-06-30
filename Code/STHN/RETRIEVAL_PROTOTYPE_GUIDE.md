# Retrieval Prototype Guide

This implements the retrieval stage requested in the professor notes from
`C:\Users\Reacher\Documents\University\Project\Retrieval`.

## Goal

Given a drone/thermal query image and a large satellite map region:

1. Precompute satellite tile embeddings.
2. Start from the estimated prior position.
3. Search tiles in an outward spiral.
4. Return Top-1, Top-3, and Top-5 matches with runtime metrics.
5. Expand Top-3 matches with overlapping shifted-grid tiles for the later IHN/STHN homography stage.

The current implementation is retrieval only. Homography refinement remains a downstream step.

## Build The Two-Grid Tile Database

Create two half-shifted grids. With the defaults, each grid is `25 x 25`, so `both`
produces up to `1250` tile records before boundary clipping.

```bash
python scripts/create_retrieval_tile_h5.py \
  --map_path /path/to/large_satellite_map.png \
  --datasets_folder datasets \
  --dataset_name retrieval_level16_two_grid \
  --split test \
  --grid_size 25 \
  --tile_size_px 512 \
  --meters_per_pixel 2.0 \
  --zoom_level 16 \
  --grids both \
  --copy_map
```

Grid IDs are written into H5 names as `@y@x@grid0@z16` and `@y@x@grid1@z16`.
The existing `BaseDataset` still reads the first two `@` fields as coordinates.

## Run Spiral Retrieval

Use the existing global evaluator with the new search strategy:

```bash
python global_pipeline/eval.py \
  --datasets_folder datasets \
  --dataset_name retrieval_level16_two_grid \
  --resume /path/to/checkpoint.pth \
  --backbone resnet18conv4 \
  --aggregation netvlad \
  --recall_values 1 3 5 \
  --search_strategy spiral \
  --spiral_search_radius_m 50000 \
  --spiral_handoff_top_n 3 \
  --spiral_handoff_neighbors 4 \
  --save_retrieval_embeddings cache/retrieval_embeddings.npz
```

If a descriptor-distance threshold is known, add:

```bash
--spiral_stop_distance 0.25
```

That enables early stopping once enough candidates have been evaluated and the
best match is below the threshold.

## Outputs

The evaluator writes these files under the normal `test/...` output directory:

- `spiral_retrieval_report.json`: Top-K accuracy, precision, recall, Top-1 error, runtime, evaluated tile counts.
- `spiral_retrieval_report_rankings.csv`: ranked database candidates for every query.
- `spiral_retrieval_report_homography_candidates.csv`: Top-3 candidates plus nearest shifted-grid overlap tiles.
- Optional `.npz` embedding cache from `--save_retrieval_embeddings`.

## Terrain Experiments

Run the same command for separate datasets or map centers covering:

- urban
- agricultural
- forest
- mountains
- desert with features
- featureless desert

Keep the output directories separate so the Top-1/Top-3/Top-5 accuracy and runtime can be compared per terrain class.

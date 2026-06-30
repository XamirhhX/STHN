# Retrieval Test Bundle

This folder is self-contained for testing the professor-requested retrieval stage
with the GeoTIFF San Francisco map.

## Contents

- `datasets/maps/satellite/20201117_BingSatellite.png` - satellite map used by the STHN loader.
- `datasets/retrieval_san_francisco_two_grid/` - retrieval H5 dataset.
  - `*_database.h5` contains two half-shifted 7 x 7 grids, 98 tile records per split.
  - `*_queries.h5` contains the GeoTIFF synthetic query chips.
- `code/global_pipeline/` - global retrieval/evaluation pipeline with spiral search support.
- `scripts/create_retrieval_tile_h5.py` - tile database generator.
- `source_geotiff/` - original GeoTIFF, preview, and metadata.

## Map Details

- Location: San Francisco preset.
- Area: 5 km x 5 km.
- Zoom: 17.
- Raster size: 4096 x 4096 px.
- Resolution: 1.220703125 m/px.
- Retrieval tile size: 512 px, about 625 m.

This is a compact local test map, not the full 25 km or 50 km professor-scale
search area.

## Example Retrieval Command

Run from this bundle folder after installing the STHN/global-pipeline Python
dependencies:

```powershell
python code\global_pipeline\eval.py ^
  --datasets_folder datasets ^
  --dataset_name retrieval_san_francisco_two_grid ^
  --resume C:\path\to\global_retrieval_checkpoint.pth ^
  --backbone resnet18conv4 ^
  --aggregation netvlad ^
  --recall_values 1 3 5 ^
  --search_strategy spiral ^
  --spiral_grid_step_m 625 ^
  --spiral_search_radius_m 5000 ^
  --spiral_handoff_top_n 3 ^
  --spiral_handoff_neighbors 4 ^
  --save_retrieval_embeddings cache\retrieval_embeddings.npz
```

The output report will be written under `test/.../spiral_retrieval_report.json`.

## Rebuild The Tile H5

```powershell
python scripts\create_retrieval_tile_h5.py ^
  --map_path datasets\maps\satellite\20201117_BingSatellite.png ^
  --datasets_folder datasets ^
  --dataset_name retrieval_san_francisco_two_grid ^
  --split test ^
  --grid_size 7 ^
  --tile_size_px 512 ^
  --tile_stride_px 512 ^
  --meters_per_pixel 1.220703125 ^
  --zoom_level 17 ^
  --grids both ^
  --copy_map
```

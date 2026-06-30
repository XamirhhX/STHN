# Professor Retrieval Benchmark - iran_lut_kaluts_50km_two_grid_cpu_20260622

## What The Professor Asked For

The notes and papers point to a retrieval/localization stage before STHN/UASTHN homography: precompute satellite-tile embeddings, search outward from an estimated prior in a spiral, return Top-1/Top-3/Top-5 candidates, measure accuracy and runtime, then hand Top-3 plus overlap tiles to the homography network.

## Dataset

- Location: Lut Desert / Kaluts, Iran
- Terrain: desert with features
- Database tiles: 1250
- Query images: 625
- Meters per pixel: 1.000000
- Tile size: 512 px / 2000.00 m

## Runtime And Accuracy

- Device: cpu (CPU)
- Descriptor: resnet18 / imagenet
- Embedding extraction: 98044.141 ms total
- Spiral search: 2299.338 ms total
- Mean search time: 3.679 ms/query
- Mean search FPS: 271.82
- Mean tiles evaluated: 1219.7
- Top-1 accuracy: 18.88%
- Top-3 accuracy: 38.24%
- Top-5 accuracy: 44.48%
- Top-1 mean center error: 13112.269 m

## Notes

A match is counted positive when the query center falls inside the retrieved tile footprint. This handles the professor's two-grid half-tile shift, where either overlapping grid can be a valid handoff candidate.

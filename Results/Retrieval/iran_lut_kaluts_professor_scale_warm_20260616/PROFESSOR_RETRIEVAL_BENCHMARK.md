# Professor Retrieval Benchmark - iran_lut_kaluts_50km_two_grid_warm

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

- Device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
- Descriptor: resnet18 / imagenet
- Embedding extraction: 10944.911 ms total
- Spiral search: 2819.930 ms total
- Mean search time: 4.512 ms/query
- Mean search FPS: 221.64
- Mean tiles evaluated: 1219.7
- Top-1 accuracy: 18.72%
- Top-3 accuracy: 38.40%
- Top-5 accuracy: 44.64%
- Top-1 mean center error: 13115.066 m

## Notes

A match is counted positive when the query center falls inside the retrieved tile footprint. This handles the professor's two-grid half-tile shift, where either overlapping grid can be a valid handoff candidate.

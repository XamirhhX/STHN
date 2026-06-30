# Professor Retrieval Benchmark - san_francisco_two_grid

## What The Professor Asked For

The notes and papers point to a retrieval/localization stage before STHN/UASTHN homography: precompute satellite-tile embeddings, search outward from an estimated prior in a spiral, return Top-1/Top-3/Top-5 candidates, measure accuracy and runtime, then hand Top-3 plus overlap tiles to the homography network.

## Dataset

- Location: San Francisco, CA
- Terrain: urban
- Database tiles: 98
- Query images: 3
- Meters per pixel: 1.220703
- Tile size: 512 px / 625.00 m

## Runtime And Accuracy

- Device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
- Descriptor: resnet18 / imagenet
- Embedding extraction: 44091.719 ms total
- Spiral search: 32.103 ms total
- Mean search time: 10.701 ms/query
- Mean search FPS: 93.45
- Mean tiles evaluated: 98.0
- Top-1 accuracy: 100.00%
- Top-3 accuracy: 100.00%
- Top-5 accuracy: 100.00%
- Top-1 mean center error: 0.000 m

## Notes

A match is counted positive when the query center falls inside the retrieved tile footprint. This handles the professor's two-grid half-tile shift, where either overlapping grid can be a valid handoff candidate.

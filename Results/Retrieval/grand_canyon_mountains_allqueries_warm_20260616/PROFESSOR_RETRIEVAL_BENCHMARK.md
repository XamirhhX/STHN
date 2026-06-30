# Professor Retrieval Benchmark - grand_canyon_all_patches_warm

## What The Professor Asked For

The notes and papers point to a retrieval/localization stage before STHN/UASTHN homography: precompute satellite-tile embeddings, search outward from an estimated prior in a spiral, return Top-1/Top-3/Top-5 candidates, measure accuracy and runtime, then hand Top-3 plus overlap tiles to the homography network.

## Dataset

- Location: Grand Canyon, AZ
- Terrain: mountains
- Database tiles: 196
- Query images: 196
- Meters per pixel: 0.610352
- Tile size: 512 px / 312.50 m

## Runtime And Accuracy

- Device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
- Descriptor: resnet18 / imagenet
- Embedding extraction: 3685.327 ms total
- Spiral search: 490.998 ms total
- Mean search time: 2.505 ms/query
- Mean search FPS: 399.19
- Mean tiles evaluated: 195.3
- Top-1 accuracy: 57.14%
- Top-3 accuracy: 68.88%
- Top-5 accuracy: 77.55%
- Top-1 mean center error: 736.843 m

## Notes

A match is counted positive when the query center falls inside the retrieved tile footprint. This handles the professor's two-grid half-tile shift, where either overlapping grid can be a valid handoff candidate.

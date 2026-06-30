# Amirhosein Hashemi – SHTN Project: “Being Lost” Simulation Using Retrieval Search 

The retrieval stage preceding the homography refinement step was evaluated. The approach: for each query image, its estimated prior location is used to initiate a spiral search across the satellite tile database. The tiles are then ranked based on ResNet feature distance, and performance is measured by whether the correct tile appears within the Top-1, Top-3, or Top-5 results.

This experiment was conducted on a CPU-based laptop setup along with another test ran on an RTX 4060 laptop GPU. The goal was to check visually and numerically how we can simulate being lost in a map while having an estimated last location. 


## Results

CPU run:

| Dataset | Database tiles | Queries | Top-1 | Top-3 | Top-5 | Mean search time | FPS | Mean tiles searched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| San Francisco urban | 98 | 36 | 97.22% | 100.00% | 100.00% | 1.084 ms/query | 922.26 | 97.97 |
| Grand Canyon mountains | 196 | 196 | 57.14% | 68.88% | 77.55% | 2.385 ms/query | 419.29 | 195.29 |
| Kaluts / Lut Desert | 1,250 | 625 | 18.88% | 38.24% | 44.48% | 3.679 ms/query | 271.82 | 1,219.70 |

RTX 4060 laptop GPU run:

| Dataset | Database tiles | Queries | Top-1 | Top-3 | Top-5 | Mean search time | FPS | Mean tiles searched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| San Francisco urban | 98 | 36 | 97.22% | 100.00% | 100.00% | 3.508 ms/query | 285.07 | 97.97 |
| Grand Canyon mountains | 196 | 196 | 57.14% | 68.88% | 77.55% | 2.505 ms/query | 399.19 | 195.29 |
| Kaluts / Lut Desert | 1,250 | 625 | 18.72% | 38.40% | 44.64% | 4.512 ms/query | 221.64 | 1,219.70 |

Embedding extraction time on CPU was separate from the search time:

| Dataset | CPU embedding time |
|---|---:|
| San Francisco urban | 82.027 s |
| Grand Canyon mountains | 23.509 s |
| Kaluts / Lut Desert | 98.044 s |

Embedding extraction time on the RTX 4060 GPU:

| Dataset | GPU embedding time |
|---|---:|
| San Francisco urban | 43.897 s |
| Grand Canyon mountains | 3.685 s |
| Kaluts / Lut Desert | 10.945 s |

The Top-K accuracy is almost the same between CPU and GPU. The small difference in Kaluts is most likely from floating-point/top-k tie behavior because many desert tiles look very similar. For search time, CPU can look faster in this small benchmark because the search is mostly vector comparison over already extracted embeddings, and CUDA launch/synchronization overhead matters. The GPU is much more useful for embedding extraction, especially on the larger datasets.

## Observation

San Francisco was the easiest case. Most queries were found correctly at Rank 1, and the only difficult example still had the correct tile at Rank 2. This is expected because urban structure has many strong visual features.

![San Francisco overview](C:/Users/Reacher/Documents/University/Project/Project/04_Results/Retrieval/san_francisco_urban_allqueries_cpu_20260622/visualizations_10_queries_20260622/san_francisco_10_query_search_overview.png)

This San Francisco example shows the only non-Top-1 case. The first retrieved tile is wrong, but the correct tile appears at Rank 2, so it is still successful for Top-3.

![San Francisco query 10](C:/Users/Reacher/Documents/University/Project/Project/04_Results/Retrieval/san_francisco_urban_allqueries_cpu_20260622/visualizations_10_queries_20260622/san_francisco_query_010_top3-hit.png)

The Kaluts desert test was much harder. The database has 1,250 candidate tiles and many desert areas look visually similar, so the ImageNet ResNet baseline often retrieves visually similar but geographically wrong tiles.

![Kaluts overview](C:/Users/Reacher/Documents/University/Project/Project/04_Results/Retrieval/iran_lut_kaluts_professor_scale_cpu_20260622/visualizations_10_queries_20260622/kaluts_10_query_search_overview.png)

This failed Kaluts example shows the main limitation: all Top-5 tiles look similar to the query, but none of them are actually the correct geographic tile.

![Kaluts query 107](C:/Users/Reacher/Documents/University/Project/Project/04_Results/Retrieval/iran_lut_kaluts_professor_scale_cpu_20260622/visualizations_10_queries_20260622/kaluts_query_107_top5-miss.png)

## Conclusion

The retrieval pipeline itself is implemented and working: it loads H5 query/database tiles, extracts ResNet embeddings, searches outward from the prior in a spiral, reports Top-1/Top-3/Top-5 accuracy and timing, and writes candidate tiles for the next homography step.

The main result is that the system works very well on urban data, but the simple ImageNet ResNet descriptor is not strong enough for difficult desert terrain. For the next step, the descriptor should be replaced with a trained satellite-thermal/global retrieval model, while keeping the same search and evaluation pipeline.

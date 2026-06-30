# CPU Retrieval Run And Implementation Explanation - 2026-06-22

This file documents the CPU rerun of the professor retrieval benchmark and maps the implementation to the exact code that performs each requested part of the test.

## Files Used

- Benchmark script: `C:\Users\Reacher\Documents\University\Project\Project\STHN-main\STHN-main\scripts\professor_retrieval_benchmark.py`
- Spiral/homography helper: `C:\Users\Reacher\Documents\University\Project\Project\STHN-main\STHN-main\global_pipeline\spiral_retrieval.py`
- CPU virtual environment: `C:\Users\Reacher\Documents\University\Project\Project\STHN-main\STHN-main\.venv_retrieval_cpu`
- Results root: `C:\Users\Reacher\Documents\University\Project\Project\04_Results\Retrieval`

The CPU environment used `torch 2.12.1+cpu`, `torchvision 0.27.1+cpu`, `h5py 3.16.0`, `numpy 2.4.4`, and `pillow 12.2.0`. `torch.cuda.is_available()` was `False`, so the measured run used CPU only.

## CPU Results

These are the same three warm-result cases from `PROFESSOR_RETRIEVAL_SUMMARY_2026-06-16.md`, rerun with `--device cpu`.

| Dataset | DB tiles | Queries | Top-1 | Top-3 | Top-5 | Mean ms/query | Median ms/query | FPS | Mean tiles | Embedding time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| San Francisco urban | 98 | 36 | 97.22% | 100.00% | 100.00% | 1.084 | 0.721 | 922.26 | 97.97 | 82.027 s |
| Grand Canyon mountains | 196 | 196 | 57.14% | 68.88% | 77.55% | 2.385 | 1.639 | 419.29 | 195.29 | 23.509 s |
| Kaluts 50 km professor-scale | 1,250 | 625 | 18.88% | 38.24% | 44.48% | 3.679 | 3.307 | 271.82 | 1,219.70 | 98.044 s |

Extra metrics written in the JSON reports:

| Dataset | Precision@1 / @3 / @5 | Recall@1 / @3 / @5 | Top-1 mean / median / max center error |
|---|---:|---:|---:|
| San Francisco urban | 97.22 / 60.19 / 44.44 | 19.44 / 36.11 / 44.44 | 41.913 / 0.000 / 625.000 m |
| Grand Canyon mountains | 57.14 / 22.96 / 15.51 | 57.14 / 68.88 / 77.55 | 736.843 / 0.000 / 4,474.319 m |
| Kaluts 50 km professor-scale | 18.88 / 16.75 / 13.25 | 4.10 / 10.79 / 14.12 | 13,112.269 / 8,602.325 / 48,846.699 m |

Output folders:

- `C:\Users\Reacher\Documents\University\Project\Project\04_Results\Retrieval\san_francisco_urban_allqueries_cpu_20260622`
- `C:\Users\Reacher\Documents\University\Project\Project\04_Results\Retrieval\grand_canyon_mountains_allqueries_cpu_20260622`
- `C:\Users\Reacher\Documents\University\Project\Project\04_Results\Retrieval\iran_lut_kaluts_professor_scale_cpu_20260622`

Each output folder contains:

- `professor_retrieval_report.json`
- `professor_retrieval_rankings.csv`
- `professor_homography_candidates.csv`
- `PROFESSOR_RETRIEVAL_BENCHMARK.md`

The Kaluts CPU Top-K values differ from the earlier GPU report by one or two query outcomes. This is consistent with CPU vs CUDA floating-point/top-k tie behavior in a visually repetitive desert dataset. The CPU JSON files are the authoritative result for this rerun.

## What Professor Wanted

The professor-requested retrieval test is: take a UAV/thermal-like query, use an estimated prior location, search satellite tiles outward from that prior, report Top-1/Top-3/Top-5 retrieval accuracy and runtime, then write candidate satellite tiles for the later homography stage.

The implementation is retrieval only. It does not run STHN/IHN homography refinement.

## 1. Command-Line Configuration

The script exposes the exact pieces needed for the test: database H5 files, query H5 files, map/metadata paths, device, descriptor model, Top-K values, spiral search settings, and homography handoff settings.

From `professor_retrieval_benchmark.py:50-88`:

```python
parser = argparse.ArgumentParser(
    description=(
        "Professor-requested retrieval benchmark: CUDA ResNet embeddings, "
        "prior-centered spiral search, Top-1/3/5 metrics, and homography handoff CSV."
    )
)
parser.add_argument("--dataset_label", required=True)
parser.add_argument("--database_h5", nargs="+", required=True)
parser.add_argument("--queries_h5", nargs="+", required=True)
parser.add_argument("--map_path", default=None, help="Satellite map used when database H5 has no image_data.")
parser.add_argument("--metadata_json", default=None, help="GeoTIFF metadata JSON for meters-per-pixel.")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
parser.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 3, 5])
parser.add_argument("--spiral_grid_step_m", type=float, default=None)
parser.add_argument("--spiral_search_radius_m", type=float, default=None)
parser.add_argument("--handoff_top_n", type=int, default=3)
parser.add_argument("--handoff_neighbors", type=int, default=4)
```

Meaning: this is where the benchmark is told to run Top-1/3/5 retrieval, whether to use `cuda` or `cpu`, how far to search, and how many candidates to hand to the homography stage.

## 2. Loading The H5 Database And Queries

The benchmark reads image names from each H5 file, parses the y/x coordinate from the name, and creates an `ImageRecord` for each database or query item.

From `professor_retrieval_benchmark.py:129-151`:

```python
with h5py.File(h5_path, "r") as h5:
    names = [decode_name(value) for value in h5["image_name"][:]]
for index, name in enumerate(names):
    y_px, x_px = parse_yx_from_name(name)
    records.append(
        ImageRecord(
            name=name,
            y_px=y_px,
            x_px=x_px,
            source="h5" if has_data else "map",
            h5_path=str(h5_path),
            h5_index=index if has_data else None,
            map_path=map_path if not has_data else None,
        )
    )
```

Meaning: the test database and queries are not manually listed; they are loaded from H5. The coordinate in each H5 `image_name` is what later supports prior-centered retrieval and correctness checks.

## 3. Loading Image Pixels

If the H5 already contains `image_data`, the benchmark reads pixels directly. If the database H5 only contains tile centers, it crops from the full satellite map using `--map_path`.

From `professor_retrieval_benchmark.py:154-172`:

```python
if record.source == "h5":
    assert record.h5_path is not None and record.h5_index is not None
    with h5py.File(record.h5_path, "r") as h5:
        array = h5["image_data"][record.h5_index]
    return Image.fromarray(array).convert("RGB")
if record.source == "map":
    assert record.map_path is not None
    half = tile_size_px // 2
    with Image.open(record.map_path) as image:
        image = image.convert("RGB")
        return image.crop(
            (
                int(round(record.x_px)) - half,
                int(round(record.y_px)) - half,
                int(round(record.x_px)) + half,
                int(round(record.y_px)) + half,
            )
        )
```

Meaning: for San Francisco, the database tiles are cropped from the map; for the Grand Canyon and Kaluts H5 files, image pixels are read from H5.

## 4. Descriptor Model

The current benchmark uses an ImageNet ResNet descriptor baseline. It removes the classification layer by replacing `fc` with `Identity`, so the model output is a feature vector.

From `professor_retrieval_benchmark.py:209-234`:

```python
if model_name == "resnet18":
    weights = models.ResNet18_Weights.DEFAULT if weights_name == "imagenet" else None
    model = models.resnet18(weights=weights)
    default_transform = weights.transforms() if weights is not None else None
elif model_name == "resnet50":
    weights = models.ResNet50_Weights.DEFAULT if weights_name == "imagenet" else None
    model = models.resnet50(weights=weights)
    default_transform = weights.transforms() if weights is not None else None

model.fc = nn.Identity()
model.eval().to(device)
```

Meaning: this is not a trained satellite-thermal retrieval checkpoint. It is a ResNet-18/ImageNet baseline descriptor, matching the saved GPU report's descriptor setting.

## 5. Embedding Extraction And CPU/GPU Selection

The script extracts database and query embeddings in batches. For the CPU run, `--device cpu` made `requested_device` equal to `cpu`, so tensors and the ResNet model stayed on CPU.

From `professor_retrieval_benchmark.py:237-269`:

```python
requested_device = device
if requested_device == "cuda" and not torch.cuda.is_available():
    requested_device = "cpu"

model, transform = make_model(model_name, weights_name, requested_device)
features: list[np.ndarray] = []
start = time.perf_counter()

with torch.inference_mode():
    for start_index in range(0, len(records), batch_size):
        batch_records = records[start_index : start_index + batch_size]
        tensors = [transform(load_image(record, tile_size_px)) for record in batch_records]
        inputs = torch.stack(tensors, dim=0).to(requested_device, non_blocking=True)
        batch_features = model(inputs)
        batch_features = F.normalize(batch_features, p=2, dim=1)
        features.append(batch_features.detach().cpu().numpy().astype(np.float32))

elapsed_ms = (time.perf_counter() - start) * 1000.0
```

Meaning: this is the "precompute satellite-tile embeddings and query embeddings" stage. `embedding_runtime_ms` in the JSON report comes from this timer.

## 6. Main Pipeline Flow

The `main()` function wires the benchmark together: resolve map scale, load records, convert coordinates, extract or load embeddings, run spiral search, compute positives/metrics, build handoff candidates, and write output files.

From `professor_retrieval_benchmark.py:619-706`:

```python
mpp = resolve_meters_per_pixel(args, args.database_h5[0])
coordinate_units = resolve_coordinate_units(args, args.database_h5[0])
tile_size_m = resolve_tile_size_m(args, args.database_h5[0], mpp)
database_records = load_records(args.database_h5, "database", args.map_path)
query_records = load_records(args.queries_h5, "queries")
database_coords = coords_m(database_records, mpp, coordinate_units)
query_coords = coords_m(query_records, mpp, coordinate_units)
prior_coords = noisy_priors(query_coords, args.spiral_prior_noise_m, args.seed)

database_features, db_ms, embedding_device = extract_embeddings(
    database_records,
    args.tile_size_px,
    args.model,
    args.weights,
    args.device,
    args.batch_size,
)
query_features, q_ms, embedding_device = extract_embeddings(
    query_records,
    args.tile_size_px,
    args.model,
    args.weights,
    args.device,
    args.batch_size,
)
embedding_ms = db_ms + q_ms

distances, predictions, stats, search_device = search_spiral_torch(
    queries_features=query_features,
    database_features=database_features,
    database_coords=database_coords,
    prior_coords=prior_coords,
    top_k=top_k,
    device=args.device,
    grid_step_m=args.spiral_grid_step_m,
    search_radius_m=args.spiral_search_radius_m,
    stop_distance=args.spiral_stop_distance,
)
positives = positives_for_queries(
    query_coords=query_coords,
    database_coords=database_coords,
    tile_size_m=tile_size_m,
    positive_radius_m=args.positive_radius_m,
)
metrics = compute_metrics(
    predictions=predictions,
    distances=distances,
    positives=positives,
    recall_values=args.recall_values,
    query_coords=query_coords,
    database_coords=database_coords,
    stats=stats,
)
metrics["embedding_runtime_ms"] = embedding_ms
metrics["embedding_device"] = embedding_device

homography_candidates = build_homography_candidate_sets(
    predictions=predictions,
    database_utms=database_coords,
    database_paths=[record.name for record in database_records],
    top_n=args.handoff_top_n,
    overlap_neighbors=args.handoff_neighbors,
)
```

Meaning: this is the exact implementation of the professor workflow: load data, embed tiles/queries, search by prior, compute retrieval metrics, and prepare candidates for later homography.

## 7. Spiral Search Around The Prior

The spiral helper starts at the nearest tile to the prior location, groups database tiles into grid cells, then visits cells ring-by-ring outward.

From `spiral_retrieval.py:48-59`:

```python
yield (0, 0)
for ring in range(1, max_ring + 1):
    for row in range(-(ring - 1), ring + 1):
        yield (row, ring)
    for col in range(ring - 1, -ring - 1, -1):
        yield (ring, col)
    for row in range(ring - 1, -ring - 1, -1):
        yield (row, -ring)
    for col in range(-ring + 1, ring + 1):
        yield (-ring, col)
```

From `spiral_retrieval.py:75-110`:

```python
distances_to_prior = np.linalg.norm(database_utms - prior_utm[None, :], axis=1)
candidate_indexes = np.arange(len(database_utms), dtype=np.int64)
radius = _positive_or_none(search_radius_m)
if radius is not None:
    candidate_indexes = candidate_indexes[distances_to_prior <= radius]

center_index = int(candidate_indexes[np.argmin(distances_to_prior[candidate_indexes])])
origin = database_utms[center_index]
step = _positive_or_none(grid_step_m) or infer_grid_step_m(database_utms[candidate_indexes])

deltas = (database_utms[candidate_indexes] - origin[None, :]) / step
cells = _round_half_away_from_zero(deltas)

for offset in spiral_offsets(max_ring):
    ordered.extend(cell_to_indexes.pop(offset, []))

return np.asarray(ordered, dtype=np.int64), center_index, step
```

Meaning: this is where the "start at the prior and evaluate outward in a spiral" requirement is implemented.

## 8. Per-Query Retrieval Search And Timing

For every query, the benchmark computes squared L2 distances between the query feature and candidate database features ordered by the spiral. The measured search time starts immediately before this per-query search and ends after `topk`.

From `professor_retrieval_benchmark.py:292-381`:

```python
db_tensor = torch.from_numpy(database_features).to(device)
query_tensor = torch.from_numpy(queries_features).to(device)

for query_index in range(len(queries_features)):
    synchronize_if_cuda(device)
    start = time.perf_counter()
    order, center_index, inferred_step = spiral_order_indices(
        database_utms=database_coords,
        prior_utm=prior_coords[query_index],
        grid_step_m=grid_step_m,
        search_radius_m=radius,
    )
    if order.size:
        order_tensor = torch.from_numpy(order).to(device)
        candidate_features = db_tensor.index_select(0, order_tensor)
        diff = candidate_features - query_tensor[query_index].unsqueeze(0)
        candidate_distances = torch.sum(diff * diff, dim=1)
        k = min(top_k, int(candidate_distances.shape[0]))
        top_distances, top_local = torch.topk(candidate_distances, k=k, largest=False)
        synchronize_if_cuda(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        top_indexes = order[top_local.detach().cpu().numpy()]
        predictions[query_index, :k] = top_indexes
        distances[query_index, :k] = top_distances.detach().cpu().numpy()
        evaluated = int(order.size)
```

Meaning: `runtime.mean_ms`, `runtime.median_ms`, `runtime.mean_fps`, and `tiles_evaluated` are search-stage metrics, not end-to-end including embedding extraction. Embedding extraction is separately reported as `embedding_runtime_ms`.

## 9. Positive Match Definition

A retrieved tile is positive if the query center falls inside the tile footprint. Optional radius matching can be added, but the saved runs used the footprint rule.

From `professor_retrieval_benchmark.py:384-401`:

```python
half_tile = tile_size_m / 2.0
for query_coord in query_coords:
    deltas = np.abs(database_coords - query_coord[None, :])
    square_hits = np.where((deltas[:, 0] <= half_tile) & (deltas[:, 1] <= half_tile))[0]
    if positive_radius_m is not None and positive_radius_m > 0:
        radius_hits = np.where(np.linalg.norm(database_coords - query_coord[None, :], axis=1) <= positive_radius_m)[0]
        hits = np.union1d(square_hits, radius_hits)
    else:
        hits = square_hits
    positives.append(hits.astype(np.int64))
```

Meaning: for the two-grid professor-scale dataset, overlapping shifted-grid tiles can both be correct because both footprints may contain the same query center.

## 10. Metrics

Top-K accuracy, precision, recall, Top-1 center error, runtime, FPS, and tile counts are computed in `compute_metrics`.

From `professor_retrieval_benchmark.py:419-434`:

```python
for top_n in sorted(set(recall_values)):
    hits = 0
    precision_values = []
    recall_values_per_query = []
    for query_index, query_predictions in enumerate(predictions):
        pred_top = [int(value) for value in query_predictions[:top_n] if int(value) >= 0]
        positive_hits = sum(1 for value in pred_top if value in positive_sets[query_index])
        if positive_hits:
            hits += 1
        precision_values.append(positive_hits / float(top_n))
        recall_values_per_query.append(
            positive_hits / float(len(positive_sets[query_index])) if positive_sets[query_index] else 0.0
        )
    accuracy_at[str(top_n)] = 100.0 * hits / query_count if query_count else 0.0
    precision_at[str(top_n)] = 100.0 * float(np.mean(precision_values)) if precision_values else 0.0
    recall_at[str(top_n)] = 100.0 * float(np.mean(recall_values_per_query)) if recall_values_per_query else 0.0
```

From `professor_retrieval_benchmark.py:446-472`:

```python
elapsed = np.asarray([stat.elapsed_ms for stat in stats], dtype=float)
evaluated = np.asarray([stat.evaluated for stat in stats], dtype=float)
return {
    "query_count": query_count,
    "accuracy_at": accuracy_at,
    "precision_at": precision_at,
    "recall_at": recall_at,
    "top1_positive_count": top1_positive,
    "top1_center_error_m": {
        "mean": float(np.mean(top1_errors)) if top1_errors else None,
        "median": float(np.median(top1_errors)) if top1_errors else None,
        "max": float(np.max(top1_errors)) if top1_errors else None,
    },
    "runtime": {
        "total_ms": float(np.sum(elapsed)) if elapsed.size else 0.0,
        "mean_ms": float(np.mean(elapsed)) if elapsed.size else 0.0,
        "median_ms": float(np.median(elapsed)) if elapsed.size else 0.0,
        "max_ms": float(np.max(elapsed)) if elapsed.size else 0.0,
        "mean_fps": float(1000.0 / np.mean(elapsed)) if elapsed.size and np.mean(elapsed) > 0 else None,
    },
    "tiles_evaluated": {
        "mean": float(np.mean(evaluated)) if evaluated.size else 0.0,
        "median": float(np.median(evaluated)) if evaluated.size else 0.0,
        "max": int(np.max(evaluated)) if evaluated.size else 0,
        "early_stop_count": int(sum(1 for stat in stats if stat.stopped_early)),
    },
}
```

Meaning: Top-K accuracy answers "does at least one correct tile appear in the first K retrieved candidates?" Precision and recall are also saved for completeness.

## 11. Homography Handoff Candidates

The benchmark expands the Top-3 retrieval results with nearest tiles from the other shifted grid. This creates a candidate list for later IHN/STHN homography refinement.

From `spiral_retrieval.py:338-360`:

```python
for query_predictions in predictions:
    expanded: list[int] = []
    for db_index in [int(i) for i in query_predictions[:top_n] if int(i) >= 0]:
        expanded.append(db_index)
        if overlap_neighbors <= 0:
            continue
        if has_shifted_grids and grid_ids[db_index] is not None:
            neighbor_pool = [
                idx for idx, grid_id in enumerate(grid_ids)
                if grid_id is not None and grid_id != grid_ids[db_index]
            ]
        else:
            neighbor_pool = [idx for idx in range(len(database_utms)) if idx != db_index]
        if not neighbor_pool:
            continue
        neighbor_pool_array = np.asarray(neighbor_pool, dtype=np.int64)
        distances = np.linalg.norm(
            database_utms[neighbor_pool_array] - database_utms[db_index][None, :],
            axis=1,
        )
        nearest = neighbor_pool_array[np.argsort(distances)[:overlap_neighbors]]
        expanded.extend(int(idx) for idx in nearest)
    candidate_sets.append(expanded)
```

Meaning: this implements "Top-3 plus overlapping/nearby shifted-grid tiles" for the downstream homography network.

## 12. Output Reports

The script writes the JSON report, ranked retrieval CSV, handoff CSV, and short Markdown report. The JSON report is the most complete machine-readable output.

From `professor_retrieval_benchmark.py:717-740`:

```python
report = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "config": {
        "dataset_label": args.dataset_label,
        "terrain": args.terrain,
        "location": args.location,
        "database_h5": [str(Path(path).resolve()) for path in args.database_h5],
        "queries_h5": [str(Path(path).resolve()) for path in args.queries_h5],
        "database_count": len(database_records),
        "query_count": len(query_records),
        "tile_size_px": args.tile_size_px,
        "tile_size_m": tile_size_m,
        "meters_per_pixel": mpp,
        "coordinate_units": coordinate_units,
        "model": args.model,
        "weights": args.weights,
        "search_device": search_device,
        "embedding_device": embedding_device,
        "gpu_name": gpu_name,
        "recall_values": args.recall_values,
        "spiral_grid_step_m": args.spiral_grid_step_m,
        "spiral_search_radius_m": args.spiral_search_radius_m,
```

Meaning: the report records exactly what dataset, device, descriptor, search radius, grid step, and metric settings were used.

## Conclusion

The implemented test matches the professor's requested retrieval/localization stage:

1. Load satellite database tiles and UAV/thermal-like query crops from H5/map files.
2. Extract ImageNet ResNet embeddings for database and queries.
3. Start at the prior query location and evaluate candidate satellite tiles in a spiral.
4. Rank by feature distance and compute Top-1/Top-3/Top-5 metrics, precision, recall, runtime, FPS, and evaluated tile counts.
5. Write Top-3 plus shifted-grid neighbor candidates for the later STHN/IHN homography step.

It is important to keep the caveat from the GPU report: this is a retrieval-system benchmark using an ImageNet ResNet baseline descriptor, not a trained satellite-thermal retrieval checkpoint from the first paper.

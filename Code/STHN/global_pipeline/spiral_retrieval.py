from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class SpiralSearchStats:
    evaluated: int
    elapsed_ms: float
    stopped_early: bool
    center_index: int | None
    inferred_grid_step_m: float | None


def _positive_or_none(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return float(value)


def infer_grid_step_m(database_utms: np.ndarray) -> float | None:
    """Infer the smallest regular grid spacing from database coordinates."""
    steps: list[float] = []
    for axis in (0, 1):
        coords = np.unique(np.round(database_utms[:, axis].astype(float), decimals=6))
        diffs = np.diff(np.sort(coords))
        diffs = diffs[diffs > 1e-6]
        if diffs.size:
            steps.append(float(np.median(diffs)))
    if not steps:
        return None
    return float(np.median(steps))


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5)).astype(int)


def spiral_offsets(max_ring: int) -> Iterable[tuple[int, int]]:
    """Yield grid offsets in the order C, east, southeast, south, ... outward."""
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


def spiral_order_indices(
    database_utms: np.ndarray,
    prior_utm: np.ndarray,
    grid_step_m: float | None = None,
    search_radius_m: float | None = None,
) -> tuple[np.ndarray, int | None, float | None]:
    """Return database indexes ordered by an outward spiral around the prior."""
    if database_utms.size == 0:
        return np.empty((0,), dtype=np.int64), None, None

    database_utms = np.asarray(database_utms, dtype=float)
    prior_utm = np.asarray(prior_utm, dtype=float)

    distances_to_prior = np.linalg.norm(database_utms - prior_utm[None, :], axis=1)
    candidate_indexes = np.arange(len(database_utms), dtype=np.int64)
    radius = _positive_or_none(search_radius_m)
    if radius is not None:
        candidate_indexes = candidate_indexes[distances_to_prior <= radius]
    if candidate_indexes.size == 0:
        return candidate_indexes, None, _positive_or_none(grid_step_m)

    center_index = int(candidate_indexes[np.argmin(distances_to_prior[candidate_indexes])])
    origin = database_utms[center_index]
    step = _positive_or_none(grid_step_m) or infer_grid_step_m(database_utms[candidate_indexes])
    if step is None:
        ordered = candidate_indexes[np.argsort(distances_to_prior[candidate_indexes])]
        return ordered.astype(np.int64), center_index, None

    deltas = (database_utms[candidate_indexes] - origin[None, :]) / step
    cells = _round_half_away_from_zero(deltas)

    cell_to_indexes: dict[tuple[int, int], list[int]] = {}
    for index, cell in zip(candidate_indexes.tolist(), cells.tolist()):
        cell_to_indexes.setdefault((int(cell[0]), int(cell[1])), []).append(index)

    for indexes in cell_to_indexes.values():
        indexes.sort(key=lambda idx: distances_to_prior[idx])

    max_ring = max(max(abs(row), abs(col)) for row, col in cell_to_indexes)
    ordered: list[int] = []
    for offset in spiral_offsets(max_ring):
        ordered.extend(cell_to_indexes.pop(offset, []))

    if cell_to_indexes:
        leftovers = [idx for indexes in cell_to_indexes.values() for idx in indexes]
        leftovers.sort(key=lambda idx: distances_to_prior[idx])
        ordered.extend(leftovers)

    return np.asarray(ordered, dtype=np.int64), center_index, step


def search_query_spiral(
    query_feature: np.ndarray,
    database_features: np.ndarray,
    database_utms: np.ndarray,
    prior_utm: np.ndarray,
    top_k: int,
    grid_step_m: float | None = None,
    search_radius_m: float | None = None,
    stop_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, SpiralSearchStats]:
    """Search one query in spiral order and return sorted Top-K indexes and scores."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    start = time.perf_counter()
    order, center_index, inferred_step = spiral_order_indices(
        database_utms=database_utms,
        prior_utm=prior_utm,
        grid_step_m=grid_step_m,
        search_radius_m=search_radius_m,
    )
    stop_distance = _positive_or_none(stop_distance)

    if order.size == 0:
        distances = np.full((top_k,), np.inf, dtype=np.float32)
        predictions = np.full((top_k,), -1, dtype=np.int64)
        stats = SpiralSearchStats(
            evaluated=0,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            stopped_early=False,
            center_index=center_index,
            inferred_grid_step_m=inferred_step,
        )
        return distances, predictions, stats

    if stop_distance is None:
        candidate_features = database_features[order]
        diff = candidate_features - query_feature[None, :]
        candidate_distances = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
        sorted_local = np.argsort(candidate_distances)[:top_k]
        selected_indexes = order[sorted_local]
        selected_distances = candidate_distances[sorted_local]
        stopped_early = False
        evaluated = int(order.size)
    else:
        evaluated_indexes: list[int] = []
        evaluated_distances: list[float] = []
        best_distance = math.inf
        stopped_early = False
        for db_index in order.tolist():
            diff = database_features[db_index] - query_feature
            distance = float(np.dot(diff, diff))
            evaluated_indexes.append(db_index)
            evaluated_distances.append(distance)
            if distance < best_distance:
                best_distance = distance
            if len(evaluated_indexes) >= top_k and best_distance <= stop_distance:
                stopped_early = True
                break
        distances_array = np.asarray(evaluated_distances, dtype=np.float32)
        indexes_array = np.asarray(evaluated_indexes, dtype=np.int64)
        sorted_local = np.argsort(distances_array)[:top_k]
        selected_indexes = indexes_array[sorted_local]
        selected_distances = distances_array[sorted_local]
        evaluated = len(evaluated_indexes)

    if selected_indexes.size < top_k:
        missing = top_k - selected_indexes.size
        selected_indexes = np.concatenate(
            [selected_indexes, np.full((missing,), -1, dtype=np.int64)]
        )
        selected_distances = np.concatenate(
            [selected_distances, np.full((missing,), np.inf, dtype=np.float32)]
        )

    stats = SpiralSearchStats(
        evaluated=evaluated,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        stopped_early=stopped_early,
        center_index=center_index,
        inferred_grid_step_m=inferred_step,
    )
    return selected_distances.astype(np.float32), selected_indexes.astype(np.int64), stats


def make_prior_utms(
    query_utms: np.ndarray,
    prior_noise_m: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Create deterministic noisy priors for offline evaluation."""
    query_utms = np.asarray(query_utms, dtype=float)
    if prior_noise_m <= 0:
        return query_utms.copy()
    rng = np.random.default_rng(seed)
    return query_utms + rng.normal(loc=0.0, scale=prior_noise_m, size=query_utms.shape)


def search_dataset_spiral(
    queries_features: np.ndarray,
    database_features: np.ndarray,
    database_utms: np.ndarray,
    prior_utms: np.ndarray,
    top_k: int,
    grid_step_m: float | None = None,
    search_radius_m: float | None = None,
    stop_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[SpiralSearchStats]]:
    distances = np.empty((len(queries_features), top_k), dtype=np.float32)
    predictions = np.empty((len(queries_features), top_k), dtype=np.int64)
    stats: list[SpiralSearchStats] = []

    for query_index, query_feature in enumerate(queries_features):
        query_distances, query_predictions, query_stats = search_query_spiral(
            query_feature=query_feature,
            database_features=database_features,
            database_utms=database_utms,
            prior_utm=prior_utms[query_index],
            top_k=top_k,
            grid_step_m=grid_step_m,
            search_radius_m=search_radius_m,
            stop_distance=stop_distance,
        )
        distances[query_index] = query_distances
        predictions[query_index] = query_predictions
        stats.append(query_stats)

    return distances, predictions, stats


def compute_ranking_metrics(
    predictions: np.ndarray,
    positives_per_query: Iterable[np.ndarray],
    recall_values: Iterable[int],
    queries_utms: np.ndarray,
    database_utms: np.ndarray,
    stats: list[SpiralSearchStats] | None = None,
) -> dict:
    positives = [set(np.asarray(p, dtype=np.int64).tolist()) for p in positives_per_query]
    query_count = len(predictions)
    recall_values = sorted(set(int(v) for v in recall_values))

    accuracy_at: dict[str, float] = {}
    precision_at: dict[str, float] = {}
    recall_at: dict[str, float] = {}
    for top_n in recall_values:
        hits = 0
        precision_values = []
        recall_values_per_query = []
        for query_index, pred in enumerate(predictions):
            pred_top_n = [int(p) for p in pred[:top_n] if int(p) >= 0]
            positive_hits = sum(1 for p in pred_top_n if p in positives[query_index])
            if positive_hits:
                hits += 1
            precision_values.append(positive_hits / float(top_n))
            if positives[query_index]:
                recall_values_per_query.append(positive_hits / float(len(positives[query_index])))
            else:
                recall_values_per_query.append(0.0)
        accuracy_at[str(top_n)] = hits / query_count * 100.0 if query_count else 0.0
        precision_at[str(top_n)] = float(np.mean(precision_values) * 100.0) if precision_values else 0.0
        recall_at[str(top_n)] = (
            float(np.mean(recall_values_per_query) * 100.0) if recall_values_per_query else 0.0
        )

    top1_errors = []
    for query_index, pred in enumerate(predictions):
        top1 = int(pred[0])
        if top1 >= 0:
            top1_errors.append(float(np.linalg.norm(queries_utms[query_index] - database_utms[top1])))

    metrics = {
        "query_count": query_count,
        "accuracy_at": accuracy_at,
        "precision_at": precision_at,
        "recall_at": recall_at,
        "top1_error_m": {
            "mean": float(np.mean(top1_errors)) if top1_errors else None,
            "median": float(np.median(top1_errors)) if top1_errors else None,
        },
    }

    if stats:
        evaluated = np.asarray([s.evaluated for s in stats], dtype=float)
        elapsed_ms = np.asarray([s.elapsed_ms for s in stats], dtype=float)
        metrics["runtime"] = {
            "total_ms": float(np.sum(elapsed_ms)),
            "mean_ms": float(np.mean(elapsed_ms)),
            "median_ms": float(np.median(elapsed_ms)),
            "max_ms": float(np.max(elapsed_ms)),
        }
        metrics["tiles_evaluated"] = {
            "mean": float(np.mean(evaluated)),
            "median": float(np.median(evaluated)),
            "max": int(np.max(evaluated)),
            "early_stop_count": int(sum(1 for s in stats if s.stopped_early)),
        }

    return metrics


def extract_grid_id(name: str) -> int | None:
    lowered = name.lower()
    match = re.search(r"(?:^|[@_\-])(?:grid|g)[=_\-]?(\d+)", lowered)
    if match:
        return int(match.group(1))
    return None


def build_homography_candidate_sets(
    predictions: np.ndarray,
    database_utms: np.ndarray,
    database_paths: list[str],
    top_n: int = 3,
    overlap_neighbors: int = 4,
) -> list[list[int]]:
    """Expand Top-N retrieval matches with nearest tiles from the other shifted grid."""
    if top_n <= 0:
        return [[] for _ in range(len(predictions))]

    grid_ids = [extract_grid_id(path) for path in database_paths]
    known_grids = {grid_id for grid_id in grid_ids if grid_id is not None}
    has_shifted_grids = len(known_grids) >= 2
    candidate_sets: list[list[int]] = []

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

    return candidate_sets


def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if name.startswith(prefix) else name


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, SpiralSearchStats):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_spiral_outputs(
    save_dir: str,
    report_name: str,
    config: dict,
    metrics: dict,
    predictions: np.ndarray,
    distances: np.ndarray,
    positives_per_query: Iterable[np.ndarray],
    query_paths: list[str],
    database_paths: list[str],
    queries_utms: np.ndarray,
    database_utms: np.ndarray,
    stats: list[SpiralSearchStats],
    homography_candidates: list[list[int]] | None = None,
) -> dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{report_name}.json")
    rankings_path = os.path.join(save_dir, f"{report_name}_rankings.csv")
    homography_path = os.path.join(save_dir, f"{report_name}_homography_candidates.csv")

    positives = [set(np.asarray(p, dtype=np.int64).tolist()) for p in positives_per_query]
    query_names = [_strip_prefix(path, "queries_") for path in query_paths]
    database_names = [_strip_prefix(path, "database_") for path in database_paths]

    with open(rankings_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "query_index",
                "query_name",
                "rank",
                "database_index",
                "database_name",
                "distance",
                "is_positive",
                "query_y",
                "query_x",
                "database_y",
                "database_x",
            ]
        )
        for query_index, query_predictions in enumerate(predictions):
            for rank, db_index in enumerate(query_predictions, start=1):
                db_index = int(db_index)
                if db_index < 0:
                    continue
                writer.writerow(
                    [
                        query_index,
                        query_names[query_index],
                        rank,
                        db_index,
                        database_names[db_index],
                        float(distances[query_index, rank - 1]),
                        db_index in positives[query_index],
                        float(queries_utms[query_index, 0]),
                        float(queries_utms[query_index, 1]),
                        float(database_utms[db_index, 0]),
                        float(database_utms[db_index, 1]),
                    ]
                )

    output_paths = {"json": json_path, "rankings_csv": rankings_path}
    if homography_candidates is not None:
        with open(homography_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "query_index",
                    "candidate_order",
                    "database_index",
                    "database_name",
                    "database_y",
                    "database_x",
                    "grid_id",
                ]
            )
            for query_index, candidates in enumerate(homography_candidates):
                for order, db_index in enumerate(candidates, start=1):
                    writer.writerow(
                        [
                            query_index,
                            order,
                            db_index,
                            database_names[db_index],
                            float(database_utms[db_index, 0]),
                            float(database_utms[db_index, 1]),
                            extract_grid_id(database_names[db_index]),
                        ]
                    )
        output_paths["homography_csv"] = homography_path

    report = {
        "config": config,
        "metrics": metrics,
        "stats": {
            "queries": len(stats),
            "mean_inferred_grid_step_m": (
                float(np.mean([s.inferred_grid_step_m for s in stats if s.inferred_grid_step_m]))
                if any(s.inferred_grid_step_m for s in stats)
                else None
            ),
        },
        "outputs": output_paths,
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, default=_json_default)

    return output_paths

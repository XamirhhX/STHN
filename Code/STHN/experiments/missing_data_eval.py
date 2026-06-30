"""
Run a missing-data/censorship robustness experiment for pretrained STHN.

The script reuses the repository's H5/map test loader, artificially masks the
satellite crop, thermal query, or both at test time, runs the pretrained model,
and saves per-variation metrics plus an error-vs-censored-area plot.

Example:
    python experiments/missing_data_eval.py ^
        --datasets_folder datasets ^
        --dataset_name satellite_0_thermalmapping_135_train ^
        --split test ^
        --two_stages ^
        --num_variations 100 ^
        --mask_target thermal ^
        --fill_modes zero half interp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import cv2
import kornia.geometry.transform as tgm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PIPELINE = ROOT / "local_pipeline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL_PIPELINE))

import datasets_4cor_img as datasets  # noqa: E402
from STHN_demo import STHN  # noqa: E402


@dataclass
class MaskMetadata:
    actual_ratio: float
    rects: list[tuple[int, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained STHN under artificial missing pixels."
    )
    parser.add_argument("--datasets_folder", default="datasets")
    parser.add_argument("--dataset_name", default="satellite_0_thermalmapping_135_train")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--model_id", default="xjh19972/STHN")
    parser.add_argument("--two_stages", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--crop_width",
        type=int,
        default=512,
        help="Thermal query center-crop width before resizing; repo eval scripts use 512.",
    )
    parser.add_argument("--num_variations", type=int, default=100)
    parser.add_argument("--min_ratio", type=float, default=0.10)
    parser.add_argument("--max_ratio", type=float, default=0.70)
    parser.add_argument(
        "--fill_modes",
        nargs="+",
        default=["zero", "half", "interp"],
        choices=["zero", "half", "interp"],
        help="Modes are cycled across variations unless only one is provided.",
    )
    parser.add_argument(
        "--mask_target",
        default="thermal",
        choices=["thermal", "satellite", "both"],
        help="Input tensor to censor. Thermal is the UAV query; satellite is the map crop.",
    )
    parser.add_argument(
        "--mask_count",
        default="random",
        choices=["1", "2", "random"],
        help="Use one rectangle, two rectangles, or randomly choose 1 or 2.",
    )
    parser.add_argument(
        "--rotation_max_deg",
        type=float,
        default=0.0,
        help="Kept for experiment logging; geometric augmentation is disabled by default.",
    )
    parser.add_argument(
        "--scale_max",
        type=float,
        default=0.0,
        help="Kept for experiment logging; geometric augmentation is disabled by default.",
    )
    parser.add_argument("--val_positive_dist_threshold", type=int, default=50)
    parser.add_argument("--prior_location_threshold", type=int, default=-1)
    parser.add_argument("--G_contrast", default="none", choices=["none", "manual", "autocontrast", "equalize"])
    parser.add_argument("--load_test_pairs", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_examples", type=int, default=6)
    parser.add_argument(
        "--uncertainty_repeats",
        type=int,
        default=1,
        help=(
            "If >1, rerun each ratio/sample with different random mask locations "
            "and log prediction variance as a test-time uncertainty proxy."
        ),
    )
    parser.add_argument("--high_error_threshold", type=float, default=50.0)
    parser.add_argument("--high_variance_threshold", type=float, default=8.0)
    parser.add_argument("--mean_pred_threshold_px", type=float, default=3.0)
    parser.add_argument("--min_gt_motion_px", type=float, default=8.0)
    parser.add_argument("--min_polygon_area_ratio", type=float, default=0.05)
    parser.add_argument("--max_polygon_area_ratio", type=float, default=6.0)
    args = parser.parse_args()

    if not 0.0 <= args.min_ratio <= args.max_ratio <= 0.95:
        raise ValueError("Expected 0 <= min_ratio <= max_ratio <= 0.95")
    if abs(args.rotation_max_deg) > 5:
        raise ValueError("Keep rotation_max_deg at 0 or within +/-5 degrees for this workflow.")
    if args.scale_max != 0:
        raise ValueError("This workflow disables scale augmentation; keep --scale_max 0.")
    if args.uncertainty_repeats < 1:
        raise ValueError("--uncertainty_repeats must be >= 1")
    return args


def make_dataset_args(args: argparse.Namespace, model: STHN) -> SimpleNamespace:
    return SimpleNamespace(
        datasets_folder=args.datasets_folder,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment="none",
        eval_model="pretrained_hf",
        perspective_max=0,
        rotate_max=0,
        resize_max=0,
        crop_width=args.crop_width,
        resize_width=model.resize_width,
        database_size=model.database_size,
        val_positive_dist_threshold=args.val_positive_dist_threshold,
        prior_location_threshold=args.prior_location_threshold,
        G_contrast=args.G_contrast,
        load_test_pairs=args.load_test_pairs,
        generate_test_pairs=False,
    )


def flow_to_four_corners(flow_gt: torch.Tensor) -> torch.Tensor:
    flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2), dtype=flow_gt.dtype)
    flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
    flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
    flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
    flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
    return flow_4cor


def choose_rect_count(mode: str, rng: np.random.Generator) -> int:
    if mode == "random":
        return int(rng.integers(1, 3))
    return int(mode)


def make_rect_mask(
    height: int,
    width: int,
    target_ratio: float,
    rect_count: int,
    rng: np.random.Generator,
    attempts: int = 25,
) -> tuple[np.ndarray, MaskMetadata]:
    target_pixels = max(1, int(round(target_ratio * height * width)))
    best_mask = None
    best_rects: list[tuple[int, int, int, int]] = []
    best_delta = float("inf")

    for _ in range(attempts):
        mask = np.zeros((height, width), dtype=bool)
        rects: list[tuple[int, int, int, int]] = []
        remaining = target_pixels

        for rect_idx in range(rect_count):
            slots_left = rect_count - rect_idx
            rect_area = max(1, remaining // slots_left)
            aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            rect_h = max(1, int(round(math.sqrt(rect_area / aspect))))
            rect_w = max(1, int(round(rect_h * aspect)))
            rect_h = min(rect_h, height)
            rect_w = min(rect_w, width)

            y0 = int(rng.integers(0, max(1, height - rect_h + 1)))
            x0 = int(rng.integers(0, max(1, width - rect_w + 1)))
            y1 = y0 + rect_h
            x1 = x0 + rect_w
            mask[y0:y1, x0:x1] = True
            rects.append((x0, y0, x1, y1))
            remaining = max(1, target_pixels - int(mask.sum()))

        delta = abs(int(mask.sum()) - target_pixels)
        if delta < best_delta:
            best_delta = delta
            best_mask = mask
            best_rects = rects

    assert best_mask is not None
    actual_ratio = float(best_mask.mean())
    return best_mask, MaskMetadata(actual_ratio=actual_ratio, rects=best_rects)


def inpaint_single_image(image: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    image_np = image.permute(1, 2, 0).detach().cpu().numpy()
    image_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    mask_uint8 = mask.astype(np.uint8) * 255
    restored = cv2.inpaint(image_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
    restored = restored.astype(np.float32) / 255.0
    return torch.from_numpy(restored).permute(2, 0, 1)


def apply_censorship(
    images: torch.Tensor,
    ratios: Iterable[float],
    fill_modes: Iterable[str],
    mask_count: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, list[MaskMetadata]]:
    masked = images.clone()
    _, _, height, width = masked.shape
    metadatas: list[MaskMetadata] = []

    for b, (ratio, fill_mode) in enumerate(zip(ratios, fill_modes)):
        rect_count = choose_rect_count(mask_count, rng)
        mask_np, metadata = make_rect_mask(height, width, float(ratio), rect_count, rng)
        mask_t = torch.from_numpy(mask_np).to(dtype=torch.bool)

        if fill_mode == "zero":
            masked[b, :, mask_t] = 0.0
        elif fill_mode == "half":
            masked[b, :, mask_t] = 0.5
        elif fill_mode == "interp":
            masked[b] = inpaint_single_image(masked[b], mask_np).to(masked.dtype)
        else:
            raise ValueError(f"Unknown fill mode: {fill_mode}")

        metadatas.append(metadata)

    return masked, metadatas


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def four_point_org(size: int, batch: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    fp = torch.zeros((batch, 2, 2, 2), dtype=dtype)
    fp[:, :, 0, 0] = torch.tensor([0, 0], dtype=dtype)
    fp[:, :, 0, 1] = torch.tensor([size - 1, 0], dtype=dtype)
    fp[:, :, 1, 0] = torch.tensor([0, size - 1], dtype=dtype)
    fp[:, :, 1, 1] = torch.tensor([size - 1, size - 1], dtype=dtype)
    return fp


def center_error(
    four_pred: torch.Tensor,
    flow_4cor: torch.Tensor,
    resize_width: int,
    alpha: float,
) -> torch.Tensor:
    batch = four_pred.shape[0]
    org_single = four_point_org(resize_width, 1, dtype=four_pred.dtype)
    org = org_single.repeat(batch, 1, 1, 1).flatten(2).permute(0, 2, 1).contiguous()
    pred = (four_pred.cpu() + org_single).flatten(2).permute(0, 2, 1).contiguous()
    gt = (flow_4cor.cpu() + org_single).flatten(2).permute(0, 2, 1).contiguous()

    H_pred = tgm.get_perspective_transform(org, pred)
    H_gt = tgm.get_perspective_transform(org, gt)
    center = torch.tensor(
        [resize_width / 2 - 0.5, resize_width / 2 - 0.5, 1.0],
        dtype=four_pred.dtype,
    ).view(1, 3, 1).repeat(batch, 1, 1)

    pred_w = torch.bmm(H_pred, center).squeeze(2)
    gt_w = torch.bmm(H_gt, center).squeeze(2)
    pred_center = pred_w[:, :2] / pred_w[:, 2].unsqueeze(1)
    gt_center = gt_w[:, :2] / gt_w[:, 2].unsqueeze(1)
    return torch.sqrt(((pred_center - gt_center) ** 2).sum(dim=1)) * alpha


def polygon_area_and_validity(
    four_pred: torch.Tensor,
    resize_width: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[dict[str, float | bool]]:
    org = four_point_org(resize_width, four_pred.shape[0], dtype=four_pred.dtype)
    corners = (four_pred.cpu() + org).numpy()
    expected_area = float(resize_width * resize_width)
    results: list[dict[str, float | bool]] = []

    for item in corners:
        pts = np.array(
            [
                [item[0, 0, 0], item[1, 0, 0]],
                [item[0, 0, 1], item[1, 0, 1]],
                [item[0, 1, 1], item[1, 1, 1]],
                [item[0, 1, 0], item[1, 1, 0]],
            ],
            dtype=np.float64,
        )
        finite = bool(np.isfinite(pts).all())
        if not finite:
            results.append({"signed_area": float("nan"), "area_ratio": float("nan"), "invalid_geometry": True})
            continue

        x = pts[:, 0]
        y = pts[:, 1]
        signed_area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        area_ratio = abs(signed_area) / expected_area
        invalid = signed_area <= 0 or area_ratio < min_area_ratio or area_ratio > max_area_ratio
        results.append(
            {
                "signed_area": signed_area,
                "area_ratio": area_ratio,
                "invalid_geometry": bool(invalid),
            }
        )
    return results


def compute_metrics(
    four_pred: torch.Tensor,
    flow_4cor: torch.Tensor,
    resize_width: int,
    database_size: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict[str, torch.Tensor | list[dict[str, float | bool]]]:
    four_pred_cpu = four_pred.detach().cpu()
    flow_4cor_cpu = flow_4cor.detach().cpu()
    alpha = database_size / resize_width

    corner_error = torch.sqrt(((flow_4cor_cpu - four_pred_cpu) ** 2).sum(dim=1))
    mace_resize_px = corner_error.mean(dim=(1, 2))
    mace_database = mace_resize_px * alpha
    ce_database = center_error(four_pred_cpu, flow_4cor_cpu, resize_width, alpha)

    pred_mag = torch.sqrt((four_pred_cpu**2).sum(dim=1)).mean(dim=(1, 2))
    gt_mag = torch.sqrt((flow_4cor_cpu**2).sum(dim=1)).mean(dim=(1, 2))
    pred_corner_std = four_pred_cpu.flatten(1).std(dim=1)
    geom = polygon_area_and_validity(four_pred_cpu, resize_width, min_area_ratio, max_area_ratio)

    return {
        "mace_resize_px": mace_resize_px,
        "mace_database": mace_database,
        "ce_database": ce_database,
        "pred_mag_px": pred_mag,
        "gt_mag_px": gt_mag,
        "pred_corner_std_px": pred_corner_std,
        "geometry": geom,
    }


def load_model(args: argparse.Namespace) -> tuple[STHN, str, torch.device]:
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    subfolder = "two_stages" if args.two_stages else "one_stage"
    model = STHN.from_pretrained(args.model_id, subfolder=subfolder)
    model = model.to(device)
    model.eval()
    return model, subfolder, device


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = ROOT / "outputs" / f"missing_data_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(exist_ok=True)
    return out


def cycle_fill_modes(fill_modes: list[str], start: int, count: int) -> list[str]:
    return [fill_modes[(start + i) % len(fill_modes)] for i in range(count)]


def plot_results(csv_path: Path, plot_path: Path) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(9, 5.5))
    for fill_mode, group in df.groupby("fill_mode"):
        plt.scatter(
            group["actual_censored_ratio"] * 100.0,
            group["mace_database"],
            s=16,
            alpha=0.55,
            label=fill_mode,
        )
        if len(group) >= 4:
            smooth = group.sort_values("actual_censored_ratio")
            smooth["rolling"] = smooth["mace_database"].rolling(
                window=max(3, min(9, len(smooth) // 3)),
                min_periods=1,
                center=True,
            ).mean()
            plt.plot(smooth["actual_censored_ratio"] * 100.0, smooth["rolling"], linewidth=2)

    plt.xlabel("Censored Area Ratio (%)")
    plt.ylabel("Network Error (MACE, database scale)")
    plt.title("STHN Missing-Data Sensitivity")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Fill")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()


def summarize(csv_path: Path, summary_path: Path, args: argparse.Namespace, subfolder: str, model: STHN) -> None:
    df = pd.read_csv(csv_path)
    summary = {
        "model_id": args.model_id,
        "subfolder": subfolder,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_rows": int(len(df)),
        "num_variations_requested": int(args.num_variations),
        "uncertainty_repeats": int(args.uncertainty_repeats),
        "mask_target": args.mask_target,
        "fill_modes": args.fill_modes,
        "resize_width": int(model.resize_width),
        "database_size": int(model.database_size),
        "mace_database_mean": float(df["mace_database"].mean()),
        "mace_database_median": float(df["mace_database"].median()),
        "mace_database_p90": float(df["mace_database"].quantile(0.90)),
        "mace_database_p95": float(df["mace_database"].quantile(0.95)),
        "ce_database_mean": float(df["ce_database"].mean()),
        "high_error_count": int(df["flag_high_error"].sum()),
        "high_variance_count": int(df["flag_high_variance"].sum()),
        "mean_like_count": int(df["flag_mean_like_prediction"].sum()),
        "invalid_geometry_count": int(df["flag_invalid_geometry"].sum()),
        "model_uncertainty_available": bool(df["model_uncertainty_available"].any()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = make_output_dir(args)
    model, subfolder, device = load_model(args)
    dataset_args = make_dataset_args(args, model)

    data_root = Path(args.datasets_folder)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {data_root}. "
            "Download or mount the STHN dataset before running this experiment."
        )

    loader = datasets.fetch_dataloader(dataset_args, split=args.split)
    if len(loader.dataset) == 0:
        raise RuntimeError("The selected split has zero samples.")

    csv_path = out_dir / "missing_data_results.csv"
    fieldnames = [
        "variation_id",
        "repeat_id",
        "dataset_index",
        "positive_index",
        "target_censored_ratio",
        "actual_censored_ratio",
        "fill_mode",
        "mask_target",
        "mask_rects",
        "mace_resize_px",
        "mace_database",
        "ce_database",
        "pred_mag_px",
        "gt_mag_px",
        "pred_corner_std_px",
        "prediction_variance_px",
        "signed_area",
        "area_ratio",
        "model_uncertainty_available",
        "model_uncertainty",
        "flag_high_error",
        "flag_high_variance",
        "flag_mean_like_prediction",
        "flag_invalid_geometry",
    ]

    ratios = np.linspace(args.min_ratio, args.max_ratio, args.num_variations)
    written = 0
    example_count = 0
    model_uncertainty_available = hasattr(model, "predict_uncertainty")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=args.num_variations, desc="missing-data variations")
        loader_iter = iter(loader)

        while written < args.num_variations:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            satellite, thermal, flow_gt, _, _, _, index, pos_index = batch
            current_b = min(satellite.shape[0], args.num_variations - written)
            satellite = satellite[:current_b]
            thermal = thermal[:current_b]
            flow_gt = flow_gt[:current_b]
            index = index[:current_b]
            pos_index = pos_index[:current_b]

            target_ratios = ratios[written : written + current_b].tolist()
            fill_modes = cycle_fill_modes(args.fill_modes, written, current_b)
            flow_4cor = flow_to_four_corners(flow_gt)

            repeat_rows: list[list[dict[str, object]]] = []
            repeat_preds: list[torch.Tensor] = []

            for repeat_id in range(args.uncertainty_repeats):
                sat_masked = satellite.clone()
                thermal_masked = thermal.clone()
                sat_meta = [MaskMetadata(actual_ratio=0.0, rects=[]) for _ in range(current_b)]
                thermal_meta = [MaskMetadata(actual_ratio=0.0, rects=[]) for _ in range(current_b)]

                if args.mask_target in {"satellite", "both"}:
                    sat_masked, sat_meta = apply_censorship(
                        sat_masked, target_ratios, fill_modes, args.mask_count, rng
                    )
                if args.mask_target in {"thermal", "both"}:
                    thermal_masked, thermal_meta = apply_censorship(
                        thermal_masked, target_ratios, fill_modes, args.mask_count, rng
                    )

                if args.mask_target == "satellite":
                    actual_ratios = [m.actual_ratio for m in sat_meta]
                    rects = [m.rects for m in sat_meta]
                elif args.mask_target == "thermal":
                    actual_ratios = [m.actual_ratio for m in thermal_meta]
                    rects = [m.rects for m in thermal_meta]
                else:
                    actual_ratios = [
                        (sat_meta[i].actual_ratio + thermal_meta[i].actual_ratio) / 2.0
                        for i in range(current_b)
                    ]
                    rects = [
                        {"satellite": sat_meta[i].rects, "thermal": thermal_meta[i].rects}
                        for i in range(current_b)
                    ]

                if example_count < args.save_examples:
                    for b in range(current_b):
                        if example_count >= args.save_examples:
                            break
                        stem = f"variation_{written + b:04d}_repeat_{repeat_id}"
                        save_tensor_image(sat_masked[b], out_dir / "examples" / f"{stem}_satellite.png")
                        save_tensor_image(thermal_masked[b], out_dir / "examples" / f"{stem}_thermal.png")
                        example_count += 1

                with torch.no_grad():
                    four_pred = model(
                        sat_masked.to(device, non_blocking=True),
                        thermal_masked.to(device, non_blocking=True),
                    )

                repeat_preds.append(four_pred.detach().cpu())
                metrics = compute_metrics(
                    four_pred,
                    flow_4cor,
                    model.resize_width,
                    model.database_size,
                    args.min_polygon_area_ratio,
                    args.max_polygon_area_ratio,
                )

                rows_for_repeat: list[dict[str, object]] = []
                for b in range(current_b):
                    high_error = bool(metrics["mace_database"][b].item() >= args.high_error_threshold)
                    mean_like = bool(
                        metrics["pred_mag_px"][b].item() <= args.mean_pred_threshold_px
                        and metrics["gt_mag_px"][b].item() >= args.min_gt_motion_px
                        and high_error
                    )
                    geom = metrics["geometry"][b]
                    rows_for_repeat.append(
                        {
                            "variation_id": written + b,
                            "repeat_id": repeat_id,
                            "dataset_index": int(index[b].item()),
                            "positive_index": int(pos_index[b].item()),
                            "target_censored_ratio": float(target_ratios[b]),
                            "actual_censored_ratio": float(actual_ratios[b]),
                            "fill_mode": fill_modes[b],
                            "mask_target": args.mask_target,
                            "mask_rects": json.dumps(rects[b]),
                            "mace_resize_px": float(metrics["mace_resize_px"][b].item()),
                            "mace_database": float(metrics["mace_database"][b].item()),
                            "ce_database": float(metrics["ce_database"][b].item()),
                            "pred_mag_px": float(metrics["pred_mag_px"][b].item()),
                            "gt_mag_px": float(metrics["gt_mag_px"][b].item()),
                            "pred_corner_std_px": float(metrics["pred_corner_std_px"][b].item()),
                            "prediction_variance_px": float("nan"),
                            "signed_area": float(geom["signed_area"]),
                            "area_ratio": float(geom["area_ratio"]),
                            "model_uncertainty_available": bool(model_uncertainty_available),
                            "model_uncertainty": float("nan"),
                            "flag_high_error": high_error,
                            "flag_high_variance": False,
                            "flag_mean_like_prediction": mean_like,
                            "flag_invalid_geometry": bool(geom["invalid_geometry"]),
                        }
                    )
                repeat_rows.append(rows_for_repeat)

            pred_stack = torch.stack(repeat_preds, dim=0)
            if args.uncertainty_repeats > 1:
                pred_variance = pred_stack.std(dim=0).flatten(1).mean(dim=1)
            else:
                pred_variance = torch.zeros(current_b)

            for rows_for_repeat in repeat_rows:
                for b, row in enumerate(rows_for_repeat):
                    variance_value = float(pred_variance[b].item())
                    row["prediction_variance_px"] = variance_value
                    row["flag_high_variance"] = bool(variance_value >= args.high_variance_threshold)
                    writer.writerow(row)

            written += current_b
            pbar.update(current_b)
        pbar.close()

    plot_path = out_dir / "error_vs_censored_ratio.png"
    summary_path = out_dir / "summary.json"
    plot_results(csv_path, plot_path)
    summarize(csv_path, summary_path, args, subfolder, model)

    print(f"Results CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

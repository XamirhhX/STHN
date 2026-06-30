"""
H5/full-dataset scaled-observation mosaic evaluation for STHN.

This is the dataset version of scaled_observation_examples_demo.py. It uses the
STHN H5 loader, creates scaled partial thermal observations, and computes true
MACE/center error from dataset ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))

import datasets_4cor_img as datasets  # noqa: E402
from missing_data_eval import (  # noqa: E402
    compute_metrics,
    flow_to_four_corners,
    load_model,
    make_dataset_args,
    save_tensor_image,
)
from scaled_observation_utils import make_scaled_observation_mosaic, scale_from_closer_percent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STHN under scaled partial thermal observations.")
    parser.add_argument("--datasets_folder", default="datasets")
    parser.add_argument("--dataset_name", default="satellite_0_thermalmapping_135_train")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--model_id", default="xjh19972/STHN")
    parser.add_argument("--two_stages", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop_width", type=int, default=512)
    parser.add_argument("--num_variations", type=int, default=100)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--closer_percent", type=float, default=20.0)
    parser.add_argument("--min_scale", type=float, default=None)
    parser.add_argument("--max_scale", type=float, default=None)
    parser.add_argument("--num_tiles", type=int, default=2)
    parser.add_argument("--path", default="diagonal", choices=["diagonal", "horizontal", "vertical", "corners", "random"])
    parser.add_argument("--jitter", type=int, default=8)
    parser.add_argument("--fill_modes", nargs="+", default=["zero", "half", "mean"], choices=["zero", "half", "mean", "value"])
    parser.add_argument("--fill_value", type=float, default=0.0)
    parser.add_argument("--blend", default="average", choices=["average", "overwrite"])
    parser.add_argument("--val_positive_dist_threshold", type=int, default=50)
    parser.add_argument("--prior_location_threshold", type=int, default=-1)
    parser.add_argument("--G_contrast", default="none", choices=["none", "manual", "autocontrast", "equalize"])
    parser.add_argument("--load_test_pairs", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_examples", type=int, default=8)
    parser.add_argument("--min_polygon_area_ratio", type=float, default=0.05)
    parser.add_argument("--max_polygon_area_ratio", type=float, default=6.0)
    return parser.parse_args()


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = ROOT / "outputs" / f"scaled_observation_h5_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    return out


def scale_for_variation(args: argparse.Namespace, idx: int) -> float:
    if args.scale is not None:
        return float(args.scale)
    if args.min_scale is not None and args.max_scale is not None:
        if args.num_variations <= 1:
            return float(args.min_scale)
        return float(np.linspace(args.min_scale, args.max_scale, args.num_variations)[idx])
    return scale_from_closer_percent(args.closer_percent)


def plot_results(csv_path: Path, out_dir: Path) -> None:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["missing_ratio"] = float(row["missing_ratio"])
            row["scale"] = float(row["scale"])
            row["mace_database"] = float(row["mace_database"])
            rows.append(row)

    plt.figure(figsize=(8, 5))
    for fill_mode in sorted(set(row["fill_mode"] for row in rows)):
        group = [row for row in rows if row["fill_mode"] == fill_mode]
        plt.scatter(
            [row["missing_ratio"] * 100 for row in group],
            [row["mace_database"] for row in group],
            label=fill_mode,
            s=22,
        )
    plt.xlabel("No-data Area Ratio (%)")
    plt.ylabel("MACE (database-scale pixels)")
    plt.title("Scaled Partial-Observation H5 Evaluation")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mace_vs_missing_ratio.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        [row["scale"] for row in rows],
        [row["mace_database"] for row in rows],
        c=[row["missing_ratio"] for row in rows],
        s=22,
        cmap="viridis",
    )
    plt.xlabel("Linear Observation Scale")
    plt.ylabel("MACE (database-scale pixels)")
    plt.title("MACE vs Simulated Observation Scale")
    plt.colorbar(label="No-data ratio")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "mace_vs_scale.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    out_dir = output_dir(args)

    model, subfolder, device = load_model(args)
    dataset_args = make_dataset_args(args, model)
    loader = datasets.fetch_dataloader(dataset_args, split=args.split)

    csv_path = out_dir / "scaled_observation_h5_results.csv"
    fields = [
        "variation_id",
        "dataset_index",
        "positive_index",
        "scale",
        "closer_percent_convention",
        "num_tiles",
        "path",
        "coverage_ratio",
        "missing_ratio",
        "fill_mode",
        "blend",
        "placements_xyxy",
        "mace_resize_px",
        "mace_database",
        "ce_database",
        "pred_mag_px",
        "gt_mag_px",
        "pred_corner_std_px",
        "signed_area",
        "area_ratio",
        "flag_invalid_geometry",
    ]

    written = 0
    example_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        loader_iter = iter(loader)
        pbar = tqdm(total=args.num_variations, desc="scaled-observation variations")

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
            flow_4cor = flow_to_four_corners(flow_gt)

            for b in range(current_b):
                variation_id = written + b
                fill_mode = args.fill_modes[variation_id % len(args.fill_modes)]
                scale = scale_for_variation(args, variation_id)
                mosaic, meta = make_scaled_observation_mosaic(
                    thermal[b : b + 1],
                    scale=scale,
                    num_tiles=args.num_tiles,
                    path=args.path,
                    fill_mode=fill_mode,
                    fill_value=args.fill_value,
                    blend=args.blend,
                    jitter=args.jitter,
                    generator=generator,
                )

                if example_count < args.save_examples:
                    save_tensor_image(mosaic[0], out_dir / "examples" / f"variation_{variation_id:04d}_thermal_mosaic.png")
                    example_count += 1

                with torch.no_grad():
                    four_pred = model(
                        satellite[b : b + 1].to(device, non_blocking=True),
                        mosaic.to(device, non_blocking=True),
                    )

                metrics = compute_metrics(
                    four_pred,
                    flow_4cor[b : b + 1],
                    model.resize_width,
                    model.database_size,
                    args.min_polygon_area_ratio,
                    args.max_polygon_area_ratio,
                )
                geom = metrics["geometry"][0]
                writer.writerow(
                    {
                        "variation_id": variation_id,
                        "dataset_index": int(index[b].item()),
                        "positive_index": int(pos_index[b].item()),
                        "scale": meta.scale,
                        "closer_percent_convention": (1.0 - meta.scale) * 100.0,
                        "num_tiles": args.num_tiles,
                        "path": args.path,
                        "coverage_ratio": meta.coverage_ratio,
                        "missing_ratio": meta.missing_ratio,
                        "fill_mode": fill_mode,
                        "blend": args.blend,
                        "placements_xyxy": json.dumps(meta.placements),
                        "mace_resize_px": float(metrics["mace_resize_px"][0].item()),
                        "mace_database": float(metrics["mace_database"][0].item()),
                        "ce_database": float(metrics["ce_database"][0].item()),
                        "pred_mag_px": float(metrics["pred_mag_px"][0].item()),
                        "gt_mag_px": float(metrics["gt_mag_px"][0].item()),
                        "pred_corner_std_px": float(metrics["pred_corner_std_px"][0].item()),
                        "signed_area": float(geom["signed_area"]),
                        "area_ratio": float(geom["area_ratio"]),
                        "flag_invalid_geometry": bool(geom["invalid_geometry"]),
                    }
                )
                written += 1
                pbar.update(1)
        pbar.close()

    plot_results(csv_path, out_dir)
    print(f"Model: {args.model_id}/{subfolder}")
    print(f"Results CSV: {csv_path}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()


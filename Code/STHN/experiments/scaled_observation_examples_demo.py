"""
Dataset-free scaled-observation mosaic test for STHN.

This is different from censoring. It simulates the professor's scale/altitude
case where a thermal observation covers a smaller footprint, so multiple smaller
observations are pasted into one standard input canvas and the uncovered regions
become no-data.

Example:
    python experiments/scaled_observation_examples_demo.py --two_stages --scale 0.8 --num_tiles 2
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))

from scaled_observation_utils import make_scaled_observation_mosaic, scale_from_closer_percent  # noqa: E402
from STHN_demo import (  # noqa: E402
    STHN,
    load_and_preprocess_satellite,
    load_and_preprocess_thermal,
    visualize_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaled partial-observation STHN smoke test.")
    parser.add_argument("--satellite", default=str(ROOT / "examples" / "img1.png"))
    parser.add_argument("--thermal", default=str(ROOT / "examples" / "img2.png"))
    parser.add_argument("--gt_image", default=str(ROOT / "examples" / "gt.png"))
    parser.add_argument("--model_id", default="xjh19972/STHN")
    parser.add_argument("--two_stages", action="store_true")
    parser.add_argument("--random_weights", action="store_true")
    parser.add_argument("--num_variations", type=int, default=100)
    parser.add_argument("--scale", type=float, default=None, help="Fixed linear footprint scale, e.g. 0.8.")
    parser.add_argument("--closer_percent", type=float, default=20.0, help="20 means scale 0.8 by convention.")
    parser.add_argument("--min_scale", type=float, default=None)
    parser.add_argument("--max_scale", type=float, default=None)
    parser.add_argument("--num_tiles", type=int, default=2)
    parser.add_argument("--path", default="diagonal", choices=["diagonal", "horizontal", "vertical", "corners", "random"])
    parser.add_argument("--jitter", type=int, default=8)
    parser.add_argument("--fill_modes", nargs="+", default=["zero", "half", "mean"], choices=["zero", "half", "mean", "value"])
    parser.add_argument("--fill_value", type=float, default=0.0)
    parser.add_argument("--blend", default="average", choices=["average", "overwrite"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = ROOT / "outputs" / f"scaled_observation_examples_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    return out


def load_model(args: argparse.Namespace, device: torch.device) -> STHN:
    if args.random_weights:
        config = {
            "resize_width": 256,
            "database_size": 1536 if args.two_stages else 512,
            "corr_level": 4 if args.two_stages else 2,
            "two_stages": bool(args.two_stages),
            "iters_lev0": 6,
            "iters_lev1": 6,
            "fine_padding": 32 if args.two_stages else 0,
        }
        model = STHN(config)
    else:
        subfolder = "two_stages" if args.two_stages else "one_stage"
        model = STHN.from_pretrained(args.model_id, subfolder=subfolder)
    return model.to(device).eval()


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def prediction_drift(pred: torch.Tensor, reference: torch.Tensor, database_size: int, resize_width: int) -> float:
    alpha = database_size / resize_width
    drift = torch.sqrt(((pred.detach().cpu() - reference.detach().cpu()) ** 2).sum(dim=1)).mean()
    return float(drift.item() * alpha)


def scale_for_variation(args: argparse.Namespace, idx: int) -> float:
    if args.scale is not None:
        return float(args.scale)
    if args.min_scale is not None and args.max_scale is not None:
        if args.num_variations <= 1:
            return float(args.min_scale)
        return float(np.linspace(args.min_scale, args.max_scale, args.num_variations)[idx])
    return scale_from_closer_percent(args.closer_percent)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    out = output_dir(args)
    model = load_model(args, device)
    satellite = load_and_preprocess_satellite(args.satellite, model.database_size)
    thermal = load_and_preprocess_thermal(args.thermal, model.resize_width)

    with torch.no_grad():
        clean_pred = model(satellite.to(device), thermal.to(device))

    visualize_result(
        satellite.to(device),
        thermal.to(device),
        clean_pred,
        model.resize_width,
        model.database_size,
        save_path=str(out / "clean_prediction.png"),
        gt_image_path=args.gt_image if Path(args.gt_image).exists() else None,
    )

    rows = []
    for idx in range(args.num_variations):
        fill_mode = args.fill_modes[idx % len(args.fill_modes)]
        scale = scale_for_variation(args, idx)
        mosaic, meta = make_scaled_observation_mosaic(
            thermal,
            scale=scale,
            num_tiles=args.num_tiles,
            path=args.path,
            fill_mode=fill_mode,
            fill_value=args.fill_value,
            blend=args.blend,
            jitter=args.jitter,
            generator=generator,
        )
        with torch.no_grad():
            pred = model(satellite.to(device), mosaic.to(device))
        drift = prediction_drift(pred, clean_pred, model.database_size, model.resize_width)
        rows.append(
            {
                "variation_id": idx,
                "scale": meta.scale,
                "closer_percent_convention": (1.0 - meta.scale) * 100.0,
                "num_tiles": args.num_tiles,
                "path": args.path,
                "coverage_ratio": meta.coverage_ratio,
                "missing_ratio": meta.missing_ratio,
                "fill_mode": fill_mode,
                "blend": args.blend,
                "placements_xyxy": repr(meta.placements),
                "prediction_drift_vs_clean": drift,
                "random_weights": bool(args.random_weights),
            }
        )
        if idx < 12:
            save_tensor_image(mosaic, out / "examples" / f"variation_{idx:03d}_thermal_mosaic.png")

    csv_path = out / "scaled_observation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = out / "drift_vs_missing_ratio.png"
    plt.figure(figsize=(8, 5))
    for fill_mode in sorted(set(row["fill_mode"] for row in rows)):
        group = [row for row in rows if row["fill_mode"] == fill_mode]
        plt.scatter(
            [row["missing_ratio"] * 100 for row in group],
            [row["prediction_drift_vs_clean"] for row in group],
            label=fill_mode,
            s=22,
        )
    plt.xlabel("No-data Area Ratio (%)")
    plt.ylabel("Prediction Drift vs Clean (database-scale pixels)")
    plt.title("Scaled Partial-Observation Mosaic Test")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    scale_plot = out / "drift_vs_scale.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(
        [row["scale"] for row in rows],
        [row["prediction_drift_vs_clean"] for row in rows],
        c=[row["missing_ratio"] for row in rows],
        s=22,
        cmap="viridis",
    )
    plt.xlabel("Linear Observation Scale")
    plt.ylabel("Prediction Drift vs Clean (database-scale pixels)")
    plt.title("Drift vs Simulated Observation Scale")
    plt.colorbar(label="No-data ratio")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(scale_plot, dpi=180)
    plt.close()

    if args.random_weights:
        print("WARNING: --random_weights was used. Results only prove the code runs.")
    print(f"Output directory: {out}")
    print(f"CSV: {csv_path}")
    print(f"Missing-ratio plot: {plot_path}")
    print(f"Scale plot: {scale_plot}")


if __name__ == "__main__":
    main()


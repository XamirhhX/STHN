"""
Dataset-free missing-data smoke test for STHN.

This uses the repository's examples/img1.png and examples/img2.png pair. Because
the examples folder does not include numeric four-corner ground truth, the script
uses the clean pretrained prediction as a pseudo-reference and reports prediction
drift under masking. This is for deployment/debugging, not for paper metrics.

Examples:
    python experiments/missing_data_examples_demo.py --two_stages
    python experiments/missing_data_examples_demo.py --random_weights
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_pipeline"))

from STHN_demo import (  # noqa: E402
    STHN,
    load_and_preprocess_satellite,
    load_and_preprocess_thermal,
    visualize_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-dataset missing-data STHN demo.")
    parser.add_argument("--satellite", default=str(ROOT / "examples" / "img1.png"))
    parser.add_argument("--thermal", default=str(ROOT / "examples" / "img2.png"))
    parser.add_argument("--gt_image", default=str(ROOT / "examples" / "gt.png"))
    parser.add_argument("--model_id", default="xjh19972/STHN")
    parser.add_argument("--two_stages", action="store_true")
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Bypass pretrained download. This only tests code execution, not model quality.",
    )
    parser.add_argument("--num_variations", type=int, default=30)
    parser.add_argument("--min_ratio", type=float, default=0.10)
    parser.add_argument("--max_ratio", type=float, default=0.70)
    parser.add_argument("--mask_target", default="thermal", choices=["thermal", "satellite", "both"])
    parser.add_argument("--fill_modes", nargs="+", default=["zero", "half", "interp"], choices=["zero", "half", "interp"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = ROOT / "outputs" / f"missing_data_examples_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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


def make_rect_mask(height: int, width: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    target = max(1, int(round(height * width * ratio)))
    rect_count = int(rng.integers(1, 3))
    mask = np.zeros((height, width), dtype=bool)
    remaining = target
    for idx in range(rect_count):
        area = max(1, remaining // (rect_count - idx))
        aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        rect_h = min(height, max(1, int(round(math.sqrt(area / aspect)))))
        rect_w = min(width, max(1, int(round(rect_h * aspect))))
        y0 = int(rng.integers(0, max(1, height - rect_h + 1)))
        x0 = int(rng.integers(0, max(1, width - rect_w + 1)))
        mask[y0 : y0 + rect_h, x0 : x0 + rect_w] = True
        remaining = max(1, target - int(mask.sum()))
    return mask


def inpaint(image: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    image_np = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    image_u8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    restored = cv2.inpaint(image_u8, mask_u8, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    return torch.from_numpy(restored).permute(2, 0, 1).unsqueeze(0)


def apply_mask(image: torch.Tensor, ratio: float, fill_mode: str, rng: np.random.Generator) -> tuple[torch.Tensor, float]:
    masked = image.clone()
    _, _, height, width = masked.shape
    mask = make_rect_mask(height, width, ratio, rng)
    mask_t = torch.from_numpy(mask).bool()

    if fill_mode == "zero":
        masked[:, :, mask_t] = 0.0
    elif fill_mode == "half":
        masked[:, :, mask_t] = 0.5
    elif fill_mode == "interp":
        masked = inpaint(masked, mask).to(dtype=image.dtype)
    else:
        raise ValueError(fill_mode)
    return masked, float(mask.mean())


def prediction_drift(pred: torch.Tensor, reference: torch.Tensor, database_size: int, resize_width: int) -> float:
    alpha = database_size / resize_width
    drift = torch.sqrt(((pred.detach().cpu() - reference.detach().cpu()) ** 2).sum(dim=1)).mean()
    return float(drift.item() * alpha)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

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

    csv_path = out / "missing_data_examples_results.csv"
    rows = []
    ratios = np.linspace(args.min_ratio, args.max_ratio, args.num_variations)
    for idx, ratio in enumerate(ratios):
        fill_mode = args.fill_modes[idx % len(args.fill_modes)]
        sat_masked = satellite.clone()
        th_masked = thermal.clone()
        actual_ratios = []

        if args.mask_target in {"satellite", "both"}:
            sat_masked, actual = apply_mask(sat_masked, float(ratio), fill_mode, rng)
            actual_ratios.append(actual)
        if args.mask_target in {"thermal", "both"}:
            th_masked, actual = apply_mask(th_masked, float(ratio), fill_mode, rng)
            actual_ratios.append(actual)

        actual_ratio = float(np.mean(actual_ratios))
        with torch.no_grad():
            pred = model(sat_masked.to(device), th_masked.to(device))

        drift = prediction_drift(pred, clean_pred, model.database_size, model.resize_width)
        rows.append(
            {
                "variation_id": idx,
                "target_censored_ratio": float(ratio),
                "actual_censored_ratio": actual_ratio,
                "fill_mode": fill_mode,
                "mask_target": args.mask_target,
                "prediction_drift_vs_clean": drift,
                "random_weights": bool(args.random_weights),
            }
        )

        if idx < 6:
            save_tensor_image(sat_masked, out / "examples" / f"variation_{idx:03d}_satellite.png")
            save_tensor_image(th_masked, out / "examples" / f"variation_{idx:03d}_thermal.png")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = out / "prediction_drift_vs_censored_ratio.png"
    plt.figure(figsize=(8, 5))
    for fill_mode in sorted(set(row["fill_mode"] for row in rows)):
        group = [row for row in rows if row["fill_mode"] == fill_mode]
        plt.scatter(
            [row["actual_censored_ratio"] * 100 for row in group],
            [row["prediction_drift_vs_clean"] for row in group],
            label=fill_mode,
            s=22,
        )
    plt.xlabel("Censored Area Ratio (%)")
    plt.ylabel("Prediction Drift vs Clean (database-scale pixels)")
    plt.title("Dataset-Free Missing-Data Smoke Test")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    if args.random_weights:
        print("WARNING: --random_weights was used. Results only prove the code runs.")
    print(f"Output directory: {out}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()

"""
Measure whether the current GPU can run pretrained STHN inference.

Run this after installing the repository dependencies:
    python scripts/check_sthn_hardware.py --two_stages --batch_size 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_pipeline"))

from STHN_demo import STHN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check STHN inference memory use.")
    parser.add_argument("--model_id", default="xjh19972/STHN")
    parser.add_argument("--two_stages", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check the NVIDIA driver and PyTorch CUDA install.")

    device = torch.device(args.device)
    subfolder = "two_stages" if args.two_stages else "one_stage"
    model = STHN.from_pretrained(args.model_id, subfolder=subfolder).to(device).eval()

    satellite = torch.rand(
        args.batch_size,
        3,
        model.database_size,
        model.database_size,
        device=device,
    )
    thermal = torch.rand(
        args.batch_size,
        3,
        model.resize_width,
        model.resize_width,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        pred = model(satellite, thermal)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    print(f"Model: {args.model_id}/{subfolder}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Input satellite: {tuple(satellite.shape)}")
    print(f"Input thermal: {tuple(thermal.shape)}")
    print(f"Output: {tuple(pred.shape)}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        total = props.total_memory / 1024**3
        print(f"GPU: {props.name}")
        print(f"VRAM total: {total:.2f} GiB")
        print(f"Peak allocated: {allocated:.2f} GiB")
        print(f"Peak reserved: {reserved:.2f} GiB")
        if reserved < total * 0.80:
            print("Result: this batch size has comfortable inference headroom.")
        else:
            print("Result: reduce --batch_size or use one-stage mode to avoid OOM.")


if __name__ == "__main__":
    main()

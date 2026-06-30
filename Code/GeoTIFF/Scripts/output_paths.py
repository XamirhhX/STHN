from __future__ import annotations

from datetime import datetime
from pathlib import Path


def folder_has_content(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def unique_output_folder(path: Path) -> Path:
    path = path.resolve()
    if not folder_has_content(path):
        return path

    suffix = timestamp_suffix()
    first_candidate = path.with_name(f"{path.name}-{suffix}")
    if not first_candidate.exists():
        return first_candidate

    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{suffix}-{index:02d}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an unused output folder near {path}.")

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(output_root: str | Path, command: str, seed: int, run_name: str | None = None) -> Path:
    stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / command / f"{stem}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def psnr_from_mse(mse: float, max_value: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


def chw_float_to_uint8(frame) -> np.ndarray:
    arr = frame.detach().float().clamp(0, 1).cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return (arr * 255.0 + 0.5).astype(np.uint8)


def encode_png_base64(frame: np.ndarray) -> str:
    import base64
    import io

    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def save_gif(path: str | Path, frames: list[np.ndarray], fps: int = 15) -> None:
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    iio.imwrite(path, frames, duration=duration_ms, loop=0)


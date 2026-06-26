from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numcodecs import Blosc
from tqdm import tqdm

from coinrun_world_model.models.vqvae import build_vqvae_from_cfg
from coinrun_world_model.utils import resolve_device


def encode_dataset(cfg: Any, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, Path]:
    checkpoint = cfg.vqvae.checkpoint
    if checkpoint is None:
        checkpoint = _latest_checkpoint("runs/vqvae")
    device = resolve_device(str(cfg.device))
    model = build_vqvae_from_cfg(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    outputs = {}
    for split in splits:
        outputs[split] = encode_split(cfg, model, split, device)
    return outputs


@torch.no_grad()
def encode_split(cfg: Any, model, split: str, device) -> Path:
    import zarr

    path = Path(cfg.data.root) / f"{split}.zarr"
    root = zarr.open_group(str(path), mode="a")
    frames = root["frames"]
    n = int(frames.shape[0])
    latent_h = int(cfg.transformer.latent_h)
    latent_w = int(cfg.transformer.latent_w)
    chunk = min(int(cfg.data.chunk_frames), n)
    if "codes" in root:
        del root["codes"]
    codes = root.create_dataset(
        "codes",
        shape=(n, latent_h, latent_w),
        chunks=(chunk, latent_h, latent_w),
        dtype="int64",
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    batch_size = int(cfg.training.batch_size)
    for start in tqdm(range(0, n, batch_size), desc=f"encode:{split}"):
        end = min(n, start + batch_size)
        batch = np.asarray(frames[start:end], dtype=np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        indices = model.encode_indices(tensor).cpu().numpy()
        codes[start:end] = indices
    root.attrs["codebook_size"] = int(cfg.vqvae.codebook_size)
    root.attrs["code_shape"] = [latent_h, latent_w]
    return path


def _latest_checkpoint(root: str | Path) -> Path:
    candidates = sorted(Path(root).glob("*/best.pt")) + sorted(Path(root).glob("*/latest.pt"))
    if not candidates:
        raise FileNotFoundError("No VQ-VAE checkpoint found; pass vqvae.checkpoint=...")
    return candidates[-1]


"""Render a side-by-side (ground truth vs. world model) rollout GIF for the README.

Seeds the transformer with real context frames from the test split, replays the
recorded action sequence through the model, and writes an upscaled GIF where the
left panel is the real episode and the right panel is the model's rollout.

Usage:
    python scripts/make_demo_gif.py \
        --config configs/default.yaml \
        --vqvae runs/vqvae/<run>/latest.pt \
        --transformer runs/transformer/<run>/best.pt \
        --out docs/demo.gif
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from coinrun_world_model.config import load_config
from coinrun_world_model.data.zarr_dataset import FrameSequenceDataset
from coinrun_world_model.evaluate import load_transformer, load_vqvae
from coinrun_world_model.utils import chw_float_to_uint8, save_gif, set_seed


def upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(frame, np.ones((factor, factor, 1), dtype=frame.dtype))


def compose(real: np.ndarray, fake: np.ndarray, label: str, scale: int, header: int) -> np.ndarray:
    real_up, fake_up = upscale(real, scale), upscale(fake, scale)
    gap = np.full((real_up.shape[0], 4, 3), 30, dtype=np.uint8)
    panel = np.concatenate([real_up, gap, fake_up], axis=1)
    canvas = Image.new("RGB", (panel.shape[1], panel.shape[0] + header), (18, 18, 24))
    canvas.paste(Image.fromarray(panel), (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"ground truth | world model   {label}", fill=(235, 235, 235))
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--vqvae", default=None, help="VQ-VAE checkpoint path")
    parser.add_argument("--transformer", default=None, help="Transformer checkpoint path")
    parser.add_argument("--out", default="docs/demo.gif")
    parser.add_argument("--clip", type=int, default=0, help="Index of the test clip to render")
    parser.add_argument("--horizon", type=int, default=16, help="Generated rollout length")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    overrides = []
    if args.vqvae:
        overrides.append(f"vqvae.checkpoint={args.vqvae}")
    if args.transformer:
        overrides.append(f"transformer.checkpoint={args.transformer}")
    cfg = load_config(args.config, overrides)
    set_seed(int(cfg.seed))
    device = torch.device(args.device)

    vqvae = load_vqvae(cfg, device)
    transformer = load_transformer(cfg, device)
    context = int(cfg.transformer.context_frames)
    dataset = FrameSequenceDataset(
        cfg.data.root,
        str(cfg.demo.split),
        context_frames=context,
        horizon=args.horizon,
        stride=1,
    )
    if len(dataset) == 0:
        raise SystemExit("No test clips long enough for the requested horizon.")
    batch = dataset[min(args.clip, len(dataset) - 1)]
    frames = batch["frames"][None].to(device)
    actions = batch["actions"][None].to(device)

    with torch.no_grad():
        context_codes = vqvae.encode_indices(frames[:, :context].flatten(0, 1)).view(1, context, -1)
        past_actions = actions[:, : max(0, context - 1)]
        future_actions = actions[:, context - 1 : context - 1 + args.horizon]
        gen_codes = transformer.generate_rollout(
            context_codes,
            past_actions,
            future_actions,
            context_frames=context,
            temperature=float(cfg.transformer.temperature),
            top_k=int(cfg.transformer.top_k),
        )
        gen_frames = vqvae.decode_indices(gen_codes.flatten(0, 1)).view(
            1, args.horizon, 3, int(cfg.env.height), int(cfg.env.width)
        )

    header, scale = 22, args.scale
    out_frames: list[np.ndarray] = []
    for t in range(context):
        real = chw_float_to_uint8(frames[0, t])
        out_frames.append(compose(real, real, f"context {t + 1}/{context}", scale, header))
    for t in range(args.horizon):
        real = chw_float_to_uint8(frames[0, context + t])
        fake = chw_float_to_uint8(gen_frames[0, t])
        out_frames.append(compose(real, fake, f"generated step {t + 1}/{args.horizon}", scale, header))

    save_gif(args.out, out_frames, fps=args.fps)
    print(f"Wrote {args.out} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()

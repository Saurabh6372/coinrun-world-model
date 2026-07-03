from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from coinrun_world_model.config import save_config
from coinrun_world_model.data.procgen import NOOP_ACTION
from coinrun_world_model.data.zarr_dataset import FrameSequenceDataset
from coinrun_world_model.eval.metrics import batch_psnr, batch_ssim, fvd_rgb_motion
from coinrun_world_model.eval.probe import probe_accuracy, train_probe
from coinrun_world_model.models.transformer import build_transformer_from_cfg
from coinrun_world_model.models.vqvae import build_vqvae_from_cfg
from coinrun_world_model.utils import (
    chw_float_to_uint8,
    make_run_dir,
    resolve_device,
    save_gif,
    set_seed,
    write_csv,
    write_json,
)


def evaluate(cfg: Any) -> Path:
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    run_dir = make_run_dir(cfg.output_root, "evaluate", int(cfg.seed))
    save_config(cfg, run_dir / "config.yaml")

    vqvae = load_vqvae(cfg, device)
    transformer = load_transformer(cfg, device)
    max_horizon = max(int(x) for x in cfg.eval.rollout_steps)
    dataset = FrameSequenceDataset(
        cfg.data.root,
        str(cfg.eval.split),
        context_frames=int(cfg.transformer.context_frames),
        horizon=max_horizon,
        stride=1,
        limit=int(cfg.eval.clips),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
    )

    probe = train_probe(cfg, device)
    step_rows: list[dict[str, float | int | str]] = []
    video_real: list[torch.Tensor] = []
    video_fake: list[torch.Tensor] = []
    probe_scores: list[float] = []
    generated_examples: list[np.ndarray] = []

    for batch in tqdm(loader, desc="evaluate"):
        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        context = int(cfg.transformer.context_frames)
        with torch.no_grad():
            context_codes = vqvae.encode_indices(frames[:, :context].flatten(0, 1)).view(
                frames.shape[0], context, -1
            )
            past_actions = actions[:, : max(0, context - 1)]
            future_actions = actions[:, context - 1 : context - 1 + max_horizon]
            generated = {
                "action_transformer": _rollout_frames(
                    cfg, vqvae, transformer, context_codes, past_actions, future_actions, context
                )
            }
            no_actions = torch.full_like(future_actions, NOOP_ACTION)
            generated["no_action_rollout"] = _rollout_frames(
                cfg, vqvae, transformer, context_codes, past_actions, no_actions, context
            )
            shuffled = _shuffle_actions(future_actions)
            generated["shuffled_action_rollout"] = _rollout_frames(
                cfg, vqvae, transformer, context_codes, past_actions, shuffled, context
            )
            for ab_context in [int(x) for x in cfg.eval.context_ablations]:
                if ab_context <= 0 or ab_context > context:
                    continue
                codes_k = context_codes[:, -ab_context:]
                if ab_context > 1:
                    actions_k = actions[:, context - ab_context : context - 1]
                else:
                    actions_k = actions[:, :0]
                generated[f"context_{ab_context}"] = _rollout_frames(
                    cfg, vqvae, transformer, codes_k, actions_k, future_actions, ab_context
                )

        real_future = frames[:, context : context + max_horizon]
        for model_name, model_frames in generated.items():
            for step in [int(x) for x in cfg.eval.rollout_steps]:
                real_step = real_future[:, step - 1]
                fake_step = model_frames[:, step - 1]
                psnr_values = batch_psnr(real_step, fake_step).cpu().numpy()
                ssim_values = batch_ssim(real_step, fake_step)
                step_rows.append(
                    {
                        "model": model_name,
                        "step": step,
                        "psnr": float(psnr_values.mean()),
                        "ssim": float(ssim_values.mean()),
                    }
                )
                if model_name == "action_transformer":
                    copy_frame = frames[:, context - 1]
                    copy_psnr = batch_psnr(real_step, copy_frame).cpu().numpy()
                    copy_ssim = batch_ssim(real_step, copy_frame)
                    step_rows.append(
                        {
                            "model": "copy_last",
                            "step": step,
                            "psnr": float(copy_psnr.mean()),
                            "ssim": float(copy_ssim.mean()),
                        }
                    )

        clip_len = min(int(cfg.eval.fvd_clip_len), max_horizon)
        video_real.append(real_future[:, :clip_len].detach().cpu())
        gen_frames = generated["action_transformer"]
        video_fake.append(gen_frames[:, :clip_len].detach().cpu())
        probe_scores.append(
            probe_accuracy(
                probe,
                frames[:, context - 1],
                gen_frames[:, 0],
                future_actions[:, 0],
            )
        )
        if len(generated_examples) < 1:
            for frame in gen_frames[0, : min(max_horizon, 16)]:
                generated_examples.append(chw_float_to_uint8(frame))

    real_videos = torch.cat(video_real, dim=0)
    fake_videos = torch.cat(video_fake, dim=0)
    fvd = fvd_rgb_motion(real_videos, fake_videos)
    summary = summarize_rows(step_rows)
    summary["fvd_rgb_motion"] = fvd
    summary["probe_generated_accuracy"] = float(np.mean(probe_scores)) if probe_scores else 0.0
    summary["counterfactual"] = counterfactual_sweep(cfg, vqvae, transformer, probe, dataset, device)

    write_csv(run_dir / "rollout_metrics.csv", step_rows)
    write_json(run_dir / "metrics_summary.json", summary)
    if generated_examples:
        save_gif(run_dir / "generated_rollout.gif", generated_examples, fps=15)
    return run_dir


@torch.no_grad()
def _rollout_frames(cfg, vqvae, transformer, context_codes, past_actions, future_actions, context):
    gen_codes = transformer.generate_rollout(
        context_codes,
        past_actions,
        future_actions,
        context_frames=context,
        temperature=float(cfg.transformer.temperature),
        top_k=int(cfg.transformer.top_k),
    )
    return vqvae.decode_indices(gen_codes.flatten(0, 1)).view(
        context_codes.shape[0],
        future_actions.shape[1],
        3,
        int(cfg.env.height),
        int(cfg.env.width),
    )


def _shuffle_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[0] > 1:
        return actions[torch.randperm(actions.shape[0], device=actions.device)]
    return torch.randint(0, 15, actions.shape, device=actions.device)


def summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["step"]))
        grouped.setdefault(key, {"psnr": [], "ssim": []})
        grouped[key]["psnr"].append(float(row["psnr"]))
        grouped[key]["ssim"].append(float(row["ssim"]))
    summary: dict[str, Any] = {"rollout": {}}
    for (model, step), values in sorted(grouped.items()):
        key = f"{model}_step_{step}"
        summary["rollout"][key] = {
            "psnr": float(np.mean(values["psnr"])),
            "ssim": float(np.mean(values["ssim"])),
        }
    return summary


@torch.no_grad()
def counterfactual_sweep(cfg, vqvae, transformer, probe, dataset, device) -> dict[str, Any]:
    if len(dataset) == 0:
        return {}
    batch = dataset[0]
    frames = batch["frames"][None].to(device)
    actions = batch["actions"][None].to(device)
    context = int(cfg.transformer.context_frames)
    context_codes = vqvae.encode_indices(frames[:, :context].flatten(0, 1)).view(1, context, -1)
    past_actions = actions[:, : max(0, context - 1)]
    out: dict[str, Any] = {}
    for action in [int(a) for a in cfg.eval.counterfactual_actions]:
        future_action = torch.tensor([[action]], device=device)
        gen_code = transformer.generate_rollout(
            context_codes,
            past_actions,
            future_action,
            context_frames=context,
            temperature=float(cfg.transformer.temperature),
            top_k=int(cfg.transformer.top_k),
        )
        gen_frame = vqvae.decode_indices(gen_code[:, 0])
        logits = probe(frames[:, context - 1], gen_frame)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        out[str(action)] = {
            "probe_pred": int(probs.argmax()),
            "probe_prob_for_action": float(probs[action]),
            "top3": [int(i) for i in np.argsort(-probs)[:3]],
        }
    return out


def load_vqvae(cfg, device):
    checkpoint = cfg.vqvae.checkpoint or _latest_checkpoint("runs/vqvae")
    model = build_vqvae_from_cfg(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def load_transformer(cfg, device):
    checkpoint = cfg.transformer.checkpoint or _latest_checkpoint("runs/transformer")
    model = build_transformer_from_cfg(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _latest_checkpoint(root: str | Path) -> Path:
    candidates = sorted(Path(root).glob("*/best.pt")) + sorted(Path(root).glob("*/latest.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {root}; pass an explicit checkpoint override")
    return candidates[-1]

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from coinrun_world_model.config import save_config, to_container
from coinrun_world_model.data.zarr_dataset import FrameDataset
from coinrun_world_model.models.vqvae import build_vqvae_from_cfg
from coinrun_world_model.utils import make_run_dir, psnr_from_mse, resolve_device, set_seed, write_json


def train_vqvae(cfg: Any) -> Path:
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    run_dir = make_run_dir(cfg.output_root, "vqvae", int(cfg.seed))
    save_config(cfg, run_dir / "config.yaml")

    train_ds = FrameDataset(cfg.data.root, "train")
    val_ds = FrameDataset(cfg.data.root, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = build_vqvae_from_cfg(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.amp) and device.type == "cuda")
    writer = _make_writer(run_dir)

    best_psnr = -float("inf")
    global_step = 0
    max_steps = cfg.training.max_steps
    total_epochs = int(cfg.training.epochs)
    for epoch in range(total_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"vqvae:epoch{epoch}")
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.training.amp) and device.type == "cuda"):
                out = model(batch)
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()

            mse = torch.mean((out.recon.detach() - batch) ** 2).item()
            psnr = psnr_from_mse(mse)
            if global_step % int(cfg.training.log_every) == 0:
                metrics = {
                    "train/loss": float(out.loss.item()),
                    "train/recon_loss": float(out.recon_loss.item()),
                    "train/vq_loss": float(out.vq_loss.item()),
                    "train/psnr": psnr,
                }
                pbar.set_postfix({k.split("/")[-1]: f"{v:.4f}" for k, v in metrics.items()})
                _log(writer, metrics, global_step)

            if global_step % int(cfg.training.val_every) == 0:
                val_metrics = evaluate_vqvae(model, val_loader, device)
                _log(writer, {f"val/{k}": v for k, v in val_metrics.items()}, global_step)
                if val_metrics["psnr"] > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    save_vqvae_checkpoint(run_dir / "best.pt", model, cfg, global_step, val_metrics)

            if global_step % int(cfg.training.save_every) == 0 and global_step > 0:
                save_vqvae_checkpoint(run_dir / f"step_{global_step}.pt", model, cfg, global_step, {})

            global_step += 1
            if max_steps is not None and global_step >= int(max_steps):
                break
        if max_steps is not None and global_step >= int(max_steps):
            break

    final_metrics = evaluate_vqvae(model, val_loader, device)
    save_vqvae_checkpoint(run_dir / "latest.pt", model, cfg, global_step, final_metrics)
    if final_metrics["psnr"] > best_psnr:
        save_vqvae_checkpoint(run_dir / "best.pt", model, cfg, global_step, final_metrics)
    write_json(run_dir / "metrics_summary.json", {"val": final_metrics, "steps": global_step})
    return run_dir


@torch.no_grad()
def evaluate_vqvae(model, loader, device) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        mse = torch.mean((out.recon - batch) ** 2, dim=(1, 2, 3))
        total_mse += float(mse.sum().item())
        total_loss += float(out.loss.item()) * batch.shape[0]
        total += batch.shape[0]
    mean_mse = total_mse / max(1, total)
    return {"loss": total_loss / max(1, total), "mse": mean_mse, "psnr": psnr_from_mse(mean_mse)}


def save_vqvae_checkpoint(path: str | Path, model, cfg: Any, step: int, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": to_container(cfg),
            "step": int(step),
            "metrics": metrics,
        },
        path,
    )


def _make_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(run_dir / "tb")
    except Exception:
        return None


def _log(writer, metrics: dict[str, float], step: int) -> None:
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)


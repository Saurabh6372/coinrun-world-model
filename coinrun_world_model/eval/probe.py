from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from coinrun_world_model.data.zarr_dataset import FrameSequenceDataset


class InverseDynamicsProbe(nn.Module):
    def __init__(self, num_actions: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_actions),
        )

    def forward(self, frame_t: torch.Tensor, frame_tp1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([frame_t, frame_tp1], dim=1))


def train_probe(cfg: Any, device) -> InverseDynamicsProbe:
    dataset = FrameSequenceDataset(
        cfg.data.root,
        "train",
        context_frames=1,
        horizon=1,
        stride=1,
        limit=max(int(cfg.eval.probe_steps) * int(cfg.eval.probe_batch_size), 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.eval.probe_batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
    )
    probe = InverseDynamicsProbe(num_actions=int(cfg.env.num_actions)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    probe.train()
    steps = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        actions = batch["actions"][:, 0].to(device)
        logits = probe(frames[:, 0], frames[:, 1])
        loss = F.cross_entropy(logits, actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        if steps >= int(cfg.eval.probe_steps):
            break
    return probe


@torch.no_grad()
def probe_accuracy(
    probe: InverseDynamicsProbe,
    frame_t: torch.Tensor,
    frame_tp1: torch.Tensor,
    actions: torch.Tensor,
) -> float:
    probe.eval()
    logits = probe(frame_t, frame_tp1)
    pred = logits.argmax(dim=1)
    return float((pred == actions).float().mean().item())


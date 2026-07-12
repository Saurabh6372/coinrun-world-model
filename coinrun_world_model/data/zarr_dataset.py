from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def open_split(data_root: str | Path, split: str):
    import zarr

    return zarr.open_group(str(Path(data_root) / f"{split}.zarr"), mode="r+")


class FrameDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str):
        self.root = open_split(data_root, split)
        self.frames = self.root["frames"]

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = np.asarray(self.frames[idx], dtype=np.float32) / 255.0
        return torch.from_numpy(frame).permute(2, 0, 1)


class IndexedFrameDataset(FrameDataset):
    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        return idx, super().__getitem__(idx)


def valid_sequence_starts(root: Any, frames_per_sequence: int, stride: int = 1) -> np.ndarray:
    dones = np.asarray(root["dones"][:], dtype=bool)
    episode_id = np.asarray(root["episode_id"][:], dtype=np.int64)
    max_start = len(dones) - frames_per_sequence
    starts: list[int] = []
    for start in range(0, max(0, max_start + 1), stride):
        end = start + frames_per_sequence
        if episode_id[start] != episode_id[end - 1]:
            continue
        if dones[start:end].any():
            continue
        starts.append(start)
    return np.asarray(starts, dtype=np.int64)


class FrameSequenceDataset(Dataset):
    """Returns frame windows and aligned actions.

    If `frames` has shape `[F, C, H, W]`, returned `actions` has length `F - 1` and
    `actions[i]` leads from `frames[i]` to `frames[i + 1]`.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        context_frames: int,
        horizon: int,
        stride: int = 1,
        limit: int | None = None,
    ):
        self.root = open_split(data_root, split)
        self.frames = self.root["frames"]
        self.actions = self.root["actions"]
        self.frames_per_sequence = int(context_frames) + int(horizon)
        starts = valid_sequence_starts(self.root, self.frames_per_sequence, stride=stride)
        if limit is not None:
            starts = starts[: int(limit)]
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.frames_per_sequence
        frames = np.asarray(self.frames[start:end], dtype=np.float32) / 255.0
        actions = np.asarray(self.actions[start : end - 1], dtype=np.int64)
        return {
            "frames": torch.from_numpy(frames).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(actions),
            "start": torch.tensor(start, dtype=torch.long),
        }


class CodeSequenceDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        context_frames: int,
        horizon: int,
        stride: int = 1,
        limit: int | None = None,
    ):
        self.root = open_split(data_root, split)
        if "codes" not in self.root:
            raise KeyError(f"{split}.zarr has no 'codes' dataset; run `wm encode` first")
        self.codes = self.root["codes"]
        self.actions = self.root["actions"]
        self.frames_per_sequence = int(context_frames) + int(horizon)
        starts = valid_sequence_starts(self.root, self.frames_per_sequence, stride=stride)
        if limit is not None:
            starts = starts[: int(limit)]
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.frames_per_sequence
        codes = np.asarray(self.codes[start:end], dtype=np.int64)
        actions = np.asarray(self.actions[start : end - 1], dtype=np.int64)
        return {
            "codes": torch.from_numpy(codes).view(codes.shape[0], -1),
            "actions": torch.from_numpy(actions),
            "start": torch.tensor(start, dtype=torch.long),
        }


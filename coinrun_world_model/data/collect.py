from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc
from tqdm import tqdm

from coinrun_world_model.config import to_container
from coinrun_world_model.data.procgen import (
    NOOP_ACTION,
    RIGHT_ACTION,
    RIGHT_UP_ACTION,
    make_procgen_env,
    reset_env,
    step_env,
)
from coinrun_world_model.utils import ensure_dir, write_json


@dataclass
class ActionSampler:
    rng: np.random.Generator
    mix: dict[str, float]
    sticky_keep_prob: float
    platformer_probs: dict[str, float]
    num_actions: int = 15
    previous_action: int = NOOP_ACTION

    def _policy_name(self) -> str:
        names = list(self.mix.keys())
        weights = np.asarray([self.mix[name] for name in names], dtype=np.float64)
        weights = weights / weights.sum()
        return str(self.rng.choice(names, p=weights))

    def sample(self) -> int:
        policy = self._policy_name()
        if policy == "uniform":
            action = int(self.rng.integers(0, self.num_actions))
        elif policy == "sticky":
            if self.rng.random() < self.sticky_keep_prob:
                action = self.previous_action
            else:
                action = int(self.rng.integers(0, self.num_actions))
        elif policy == "platformer_biased":
            action = self._sample_platformer()
        else:
            action = NOOP_ACTION
        self.previous_action = action
        return action

    def _sample_platformer(self) -> int:
        explicit = {int(k): float(v) for k, v in self.platformer_probs.items() if k != "other"}
        other_weight = float(self.platformer_probs.get("other", 0.0))
        remaining = [a for a in range(self.num_actions) if a not in explicit]
        actions = list(explicit.keys()) + remaining
        weights = list(explicit.values()) + [other_weight / max(1, len(remaining))] * len(remaining)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        return int(self.rng.choice(actions, p=weights))


def collect_dataset(cfg: Any, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, Path]:
    root = ensure_dir(cfg.data.root)
    manifest: dict[str, Any] = {"splits": {}, "config": to_container(cfg)}
    outputs: dict[str, Path] = {}
    for split in splits:
        split_path = root / f"{split}.zarr"
        collect_split(cfg, split, split_path)
        outputs[split] = split_path
        manifest["splits"][split] = str(split_path)
    write_json(root / "manifest.json", manifest)
    return outputs


def collect_split(cfg: Any, split: str, output_path: str | Path) -> Path:
    import zarr

    split_cfg = cfg.env.splits[split]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    total = int(split_cfg.total_frames)
    chunk = min(int(cfg.data.chunk_frames), total)
    height, width, channels = int(cfg.env.height), int(cfg.env.width), int(cfg.env.channels)

    root = zarr.open_group(str(output_path), mode="w")
    frames = root.create_dataset(
        "frames",
        shape=(total, height, width, channels),
        chunks=(chunk, height, width, channels),
        dtype="uint8",
        compressor=compressor,
    )
    actions = root.create_dataset("actions", shape=(total,), chunks=(chunk,), dtype="int64")
    rewards = root.create_dataset("rewards", shape=(total,), chunks=(chunk,), dtype="float32")
    dones = root.create_dataset("dones", shape=(total,), chunks=(chunk,), dtype="bool")
    episode_id = root.create_dataset("episode_id", shape=(total,), chunks=(chunk,), dtype="int64")
    level_seed = root.create_dataset("level_seed", shape=(total,), chunks=(chunk,), dtype="int64")
    step_ds = root.create_dataset("step", shape=(total,), chunks=(chunk,), dtype="int64")

    root.attrs.update(
        {
            "split": split,
            "env_name": str(cfg.env.name),
            "start_level": int(split_cfg.start_level),
            "num_levels": int(split_cfg.num_levels),
            "total_frames": total,
            "episode_cap": int(cfg.data.episode_cap),
            "action_alignment": "actions[t] is applied after frames[t] and leads to frames[t+1]",
        }
    )

    rng = np.random.default_rng(int(cfg.seed) + {"train": 0, "val": 1, "test": 2}.get(split, 3))
    sampler = ActionSampler(
        rng=rng,
        mix=dict(cfg.data.action_mix),
        sticky_keep_prob=float(cfg.data.sticky_keep_prob),
        platformer_probs=dict(cfg.data.platformer_probs),
        num_actions=int(cfg.env.num_actions),
        previous_action=RIGHT_ACTION if split == "train" else NOOP_ACTION,
    )

    env = make_procgen_env(cfg.env, split_cfg)
    obs = reset_env(env)
    ep_id = 0
    ep_step = 0
    try:
        for idx in tqdm(range(total), desc=f"collect:{split}"):
            action = sampler.sample()
            frames[idx] = obs
            actions[idx] = action
            episode_id[idx] = ep_id
            n_levels = max(1, int(split_cfg.num_levels))
            level_seed[idx] = int(split_cfg.start_level) + (ep_id % n_levels)
            step_ds[idx] = ep_step

            obs, reward, done, _info = step_env(env, action)
            rewards[idx] = reward
            done = bool(done or (ep_step + 1) >= int(cfg.data.episode_cap))
            dones[idx] = done
            ep_step += 1
            if done:
                ep_id += 1
                ep_step = 0
                obs = reset_env(env)
                sampler.previous_action = RIGHT_UP_ACTION if rng.random() < 0.25 else NOOP_ACTION
    finally:
        env.close()

    return output_path


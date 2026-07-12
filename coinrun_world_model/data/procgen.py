from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_COMBOS: list[tuple[str, ...]] = [
    ("LEFT", "DOWN"),
    ("LEFT",),
    ("LEFT", "UP"),
    ("DOWN",),
    (),
    ("UP",),
    ("RIGHT", "DOWN"),
    ("RIGHT",),
    ("RIGHT", "UP"),
    ("D",),
    ("A",),
    ("W",),
    ("S",),
    ("Q",),
    ("E",),
]

ACTION_NAMES = ["+".join(combo) if combo else "NOOP" for combo in ACTION_COMBOS]
NOOP_ACTION = 4
LEFT_ACTION = 1
RIGHT_ACTION = 7
RIGHT_UP_ACTION = 8

KEY_TO_ACTION = {
    "ArrowLeft": LEFT_ACTION,
    "ArrowRight": RIGHT_ACTION,
    "ArrowUp": 5,
    "ArrowDown": 3,
    "KeyA": 10,
    "KeyD": 9,
    "KeyW": 11,
    "KeyS": 12,
    "KeyQ": 13,
    "KeyE": 14,
    "Space": NOOP_ACTION,
}


@dataclass(frozen=True)
class SplitSpec:
    start_level: int
    num_levels: int
    total_frames: int


def make_procgen_env(env_cfg: Any, split_cfg: Any):
    import gym
    import procgen  # noqa: F401  # registers procgen envs with gym

    env_id = f"procgen:procgen-{env_cfg.name}-v0"
    kwargs = {
        "start_level": int(split_cfg.start_level),
        "num_levels": int(split_cfg.num_levels),
        "distribution_mode": str(env_cfg.distribution_mode),
        "center_agent": bool(env_cfg.center_agent),
        "use_backgrounds": bool(env_cfg.use_backgrounds),
        "render_mode": str(env_cfg.render_mode),
    }
    return gym.make(env_id, **kwargs)


def unwrap_obs(obs) -> np.ndarray:
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        obs = obs.get("rgb", next(iter(obs.values())))
    arr = np.asarray(obs)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[0] == 3 and arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def reset_env(env) -> np.ndarray:
    result = env.reset()
    return unwrap_obs(result)


def step_env(env, action: int):
    result = env.step(int(action))
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = result
    return unwrap_obs(obs), float(reward), bool(done), info


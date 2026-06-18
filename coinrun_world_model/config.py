from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path = "configs/default.yaml", overrides: Iterable[str] = ()) -> DictConfig:
    """Load a Hydra/OmegaConf YAML config with optional dotlist overrides."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.set_struct(cfg, False)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def to_container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


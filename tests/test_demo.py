import importlib.util
from pathlib import Path

import pytest

from coinrun_world_model.demo.server import create_app

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("omegaconf") is None, reason="omegaconf not installed"
)


def test_demo_health_app_creation(tmp_path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "seed": 1,
            "device": "cpu",
            "data": {"root": str(tmp_path / "missing")},
            "env": {"height": 64, "width": 64, "num_actions": 15},
            "vqvae": {"checkpoint": None, "codebook_size": 32, "code_dim": 16, "hidden_channels": 32, "beta": 0.25},
            "transformer": {
                "checkpoint": None,
                "context_frames": 2,
                "latent_h": 8,
                "latent_w": 8,
                "layers": 1,
                "heads": 2,
                "width": 64,
                "dropout": 0.0,
                "max_seq_len": 256,
                "temperature": 1.0,
                "top_k": 5,
                "condition_actions": True,
            },
            "demo": {"split": "test", "mock_if_missing": True, "show_context_frames": 2},
        }
    )
    app = create_app(cfg)
    assert app.title == "CoinRun World Model Demo"
    assert Path(__file__).exists()

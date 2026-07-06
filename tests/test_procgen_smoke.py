import importlib.util

import pytest

from coinrun_world_model.data.procgen import ACTION_COMBOS

pytestmark = pytest.mark.skipif(importlib.util.find_spec("procgen") is None, reason="procgen not installed")


def test_procgen_coinrun_smoke():
    import gym
    import procgen  # noqa: F401

    env = gym.make("procgen:procgen-coinrun-v0", distribution_mode="easy", start_level=0, num_levels=1)
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        assert obs.shape == (64, 64, 3)
        assert env.action_space.n == 15
        assert ACTION_COMBOS[4] == ()
        assert ACTION_COMBOS[1] == ("LEFT",)
        assert ACTION_COMBOS[7] == ("RIGHT",)
        assert ACTION_COMBOS[8] == ("RIGHT", "UP")
    finally:
        env.close()


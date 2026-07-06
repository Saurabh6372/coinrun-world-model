import importlib.util

import numpy as np
import pytest

from coinrun_world_model.data.zarr_dataset import FrameSequenceDataset, valid_sequence_starts

pytestmark = pytest.mark.skipif(importlib.util.find_spec("zarr") is None, reason="zarr not installed")


def test_valid_sequence_starts_do_not_cross_dones(tmp_path):
    import zarr

    path = tmp_path / "train.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("frames", data=np.zeros((8, 64, 64, 3), dtype=np.uint8))
    root.create_dataset("actions", data=np.arange(8, dtype=np.int64))
    root.create_dataset("rewards", data=np.zeros((8,), dtype=np.float32))
    root.create_dataset("dones", data=np.array([False, False, True, False, False, False, False, False]))
    root.create_dataset("episode_id", data=np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64))
    root.create_dataset("level_seed", data=np.zeros((8,), dtype=np.int64))
    root.create_dataset("step", data=np.arange(8, dtype=np.int64))

    starts = valid_sequence_starts(root, frames_per_sequence=3)
    assert 0 not in starts
    assert 1 not in starts
    assert set(starts.tolist()) == {3, 4, 5}


def test_sequence_action_alignment(tmp_path):
    import zarr

    data_root = tmp_path
    root = zarr.open_group(str(data_root / "train.zarr"), mode="w")
    root.create_dataset("frames", data=np.zeros((5, 64, 64, 3), dtype=np.uint8))
    root.create_dataset("actions", data=np.array([10, 11, 12, 13, 14], dtype=np.int64))
    root.create_dataset("rewards", data=np.zeros((5,), dtype=np.float32))
    root.create_dataset("dones", data=np.zeros((5,), dtype=bool))
    root.create_dataset("episode_id", data=np.zeros((5,), dtype=np.int64))
    root.create_dataset("level_seed", data=np.zeros((5,), dtype=np.int64))
    root.create_dataset("step", data=np.arange(5, dtype=np.int64))

    ds = FrameSequenceDataset(data_root, "train", context_frames=2, horizon=2)
    item = ds[0]
    assert item["frames"].shape == (4, 3, 64, 64)
    assert item["actions"].tolist() == [10, 11, 12]


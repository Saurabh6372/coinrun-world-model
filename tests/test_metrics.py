import numpy as np
import torch

from coinrun_world_model.eval.metrics import batch_psnr, batch_ssim, fvd_rgb_motion


def test_identical_frame_metrics_are_high():
    x = torch.rand(2, 3, 64, 64)
    psnr = batch_psnr(x, x)
    ssim = batch_ssim(x, x)
    assert torch.isfinite(psnr).all()
    assert float(psnr.min()) > 100
    assert np.allclose(ssim, 1.0)


def test_fvd_rgb_motion_runs_on_small_video_batch():
    real = torch.rand(3, 4, 3, 64, 64)
    fake = real.clone()
    score = fvd_rgb_motion(real, fake)
    assert abs(score) < 1e-5


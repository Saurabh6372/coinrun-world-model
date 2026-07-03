from __future__ import annotations

import numpy as np
import torch
from scipy import linalg
from skimage.metrics import structural_similarity


def batch_psnr(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((real.float() - fake.float()) ** 2, dim=tuple(range(1, real.ndim)))
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-12))


def batch_ssim(real: torch.Tensor, fake: torch.Tensor) -> np.ndarray:
    real_np = real.detach().float().clamp(0, 1).cpu().numpy()
    fake_np = fake.detach().float().clamp(0, 1).cpu().numpy()
    values = []
    for r, f in zip(real_np, fake_np, strict=True):
        r_hwc = np.transpose(r, (1, 2, 0))
        f_hwc = np.transpose(f, (1, 2, 0))
        values.append(
            structural_similarity(
                r_hwc,
                f_hwc,
                channel_axis=-1,
                data_range=1.0,
            )
        )
    return np.asarray(values, dtype=np.float64)


def video_features_rgb_motion(videos: torch.Tensor, output_size: int = 4) -> np.ndarray:
    """Deterministic lightweight video features for an FVD-compatible Fréchet distance.

    Shape: `[B, T, C, H, W]`, values in `[0, 1]`.
    """
    videos = videos.detach().float().clamp(0, 1).cpu()
    pooled = torch.nn.functional.interpolate(
        videos.flatten(0, 1),
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    ).view(videos.shape[0], videos.shape[1], videos.shape[2], output_size, output_size)
    rgb_mean = pooled.mean(dim=(1, 3, 4))
    rgb_std = pooled.std(dim=(1, 3, 4))
    first = pooled[:, 0].flatten(1)
    last = pooled[:, -1].flatten(1)
    motion = (pooled[:, 1:] - pooled[:, :-1]).abs()
    motion_mean = motion.mean(dim=(1, 3, 4))
    motion_std = motion.std(dim=(1, 3, 4))
    flat_motion = torch.nn.functional.avg_pool2d(
        motion.flatten(0, 1), kernel_size=4, stride=4
    ).view(videos.shape[0], -1)
    features = torch.cat([rgb_mean, rgb_std, motion_mean, motion_std, first, last, flat_motion], dim=1)
    return features.numpy().astype(np.float64)


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)
    if sigma1.ndim == 0:
        sigma1 = np.eye(real_features.shape[1]) * float(sigma1)
    if sigma2.ndim == 0:
        sigma2 = np.eye(fake_features.shape[1]) * float(sigma2)
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def fvd_rgb_motion(real_videos: torch.Tensor, fake_videos: torch.Tensor) -> float:
    return frechet_distance(video_features_rgb_motion(real_videos), video_features_rgb_motion(fake_videos))

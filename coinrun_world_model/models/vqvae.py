from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, c)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.beta * commitment_loss
        z_q = z_e + (z_q - z_e).detach()
        return z_q, indices.view(b, h, w), loss


@dataclass
class VQVAEOutput:
    recon: torch.Tensor
    indices: torch.Tensor
    vq_loss: torch.Tensor
    recon_loss: torch.Tensor
    loss: torch.Tensor


class VQVAE(nn.Module):
    def __init__(
        self,
        codebook_size: int = 512,
        code_dim: int = 128,
        hidden_channels: int = 128,
        beta: float = 0.25,
    ):
        super().__init__()
        h = int(hidden_channels)
        d = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.code_dim = d
        self.encoder = nn.Sequential(
            nn.Conv2d(3, h // 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(h // 2, h, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(h, h, 4, stride=2, padding=1),
            ResidualBlock(h),
            ResidualBlock(h),
            nn.GroupNorm(8, h),
            nn.SiLU(),
            nn.Conv2d(h, d, 1),
        )
        self.quantizer = VectorQuantizer(codebook_size, d, beta)
        self.decoder = nn.Sequential(
            nn.Conv2d(d, h, 1),
            ResidualBlock(h),
            ResidualBlock(h),
            nn.GroupNorm(8, h),
            nn.SiLU(),
            nn.ConvTranspose2d(h, h, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(h, h // 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(h // 2, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        return self.quantizer(z_e)

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        _z_q, indices, _loss = self.encode(x)
        return indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 2:
            side = int(indices.shape[1] ** 0.5)
            indices = indices.view(indices.shape[0], side, side)
        z_q = self.quantizer.embedding(indices.long())
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)

    def forward(self, x: torch.Tensor) -> VQVAEOutput:
        z_q, indices, vq_loss = self.encode(x)
        recon = self.decode(z_q)
        l1 = F.l1_loss(recon, x)
        mse = F.mse_loss(recon, x)
        recon_loss = l1 + mse
        loss = recon_loss + vq_loss
        return VQVAEOutput(recon=recon, indices=indices, vq_loss=vq_loss, recon_loss=recon_loss, loss=loss)


def build_vqvae_from_cfg(cfg) -> VQVAE:
    return VQVAE(
        codebook_size=int(cfg.vqvae.codebook_size),
        code_dim=int(cfg.vqvae.code_dim),
        hidden_channels=int(cfg.vqvae.hidden_channels),
        beta=float(cfg.vqvae.beta),
    )


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from coinrun_world_model.data.procgen import NOOP_ACTION


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    tokens: torch.Tensor


def build_token_stream(
    codes: torch.Tensor,
    actions: torch.Tensor,
    codebook_size: int,
    condition_actions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave `[frame codes..., action, next frame codes...]`.

    `codes`: `[B, F, L]`; `actions`: `[B, F - 1]`.
    Returns tokens and frame ids, where action tokens use frame id `-1`.
    """
    if codes.ndim != 3:
        raise ValueError(f"codes must have shape [B,F,L], got {tuple(codes.shape)}")
    b, frames, latent_tokens = codes.shape
    if actions.shape != (b, frames - 1):
        raise ValueError(
            f"actions must have shape {(b, frames - 1)}, got {tuple(actions.shape)}"
        )
    pieces = []
    frame_ids = []
    for frame_idx in range(frames):
        pieces.append(codes[:, frame_idx].long())
        frame_ids.append(torch.full((b, latent_tokens), frame_idx, device=codes.device, dtype=torch.long))
        if frame_idx < frames - 1:
            action = actions[:, frame_idx].long()
            if not condition_actions:
                action = torch.full_like(action, NOOP_ACTION)
            pieces.append(action[:, None] + int(codebook_size))
            frame_ids.append(torch.full((b, 1), -1, device=codes.device, dtype=torch.long))
    return torch.cat(pieces, dim=1), torch.cat(frame_ids, dim=1)


class ActionConditionedTransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        num_actions: int = 15,
        latent_tokens: int = 64,
        layers: int = 6,
        heads: int = 8,
        width: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        condition_actions: bool = True,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.num_actions = int(num_actions)
        self.latent_tokens = int(latent_tokens)
        self.vocab_size = self.codebook_size + self.num_actions
        self.max_seq_len = int(max_seq_len)
        self.condition_actions = bool(condition_actions)
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, self.vocab_size)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, codes: torch.Tensor, actions: torch.Tensor) -> TransformerOutput:
        tokens, frame_ids = build_token_stream(
            codes,
            actions,
            codebook_size=self.codebook_size,
            condition_actions=self.condition_actions,
        )
        logits = self.forward_tokens(tokens)
        loss = self.loss_from_tokens(logits, tokens, frame_ids)
        return TransformerOutput(logits=logits, loss=loss, tokens=tokens)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_seq_len:
            tokens = tokens[:, -self.max_seq_len :]
        seq_len = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding[:, :seq_len]
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        x = self.blocks(x, mask=causal_mask)
        return self.head(self.norm(x))

    def loss_from_tokens(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        frame_ids: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[1] > logits.shape[1]:
            tokens = tokens[:, -logits.shape[1] :]
            frame_ids = frame_ids[:, -logits.shape[1] :]
        pred = logits[:, :-1].reshape(-1, self.vocab_size)
        target = tokens[:, 1:].reshape(-1)
        target_frame_ids = frame_ids[:, 1:].reshape(-1)
        mask = (target < self.codebook_size) & (target_frame_ids >= 1)
        if not mask.any():
            return pred.sum() * 0.0
        return F.cross_entropy(pred[mask], target[mask])

    @torch.no_grad()
    def generate_next_frame(
        self,
        context_codes: torch.Tensor,
        context_actions: torch.Tensor,
        next_action: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        """Generate one latent frame conditioned on prior frames and the next action."""
        self.eval()
        b = context_codes.shape[0]
        device = context_codes.device
        if context_actions.numel() == 0:
            actions = next_action.view(b, 1)
        else:
            actions = torch.cat([context_actions.long(), next_action.view(b, 1).long()], dim=1)
        placeholder = torch.zeros(b, 1, self.latent_tokens, dtype=torch.long, device=device)
        codes_with_placeholder = torch.cat([context_codes.long(), placeholder], dim=1)
        tokens, frame_ids = build_token_stream(
            codes_with_placeholder,
            actions,
            codebook_size=self.codebook_size,
            condition_actions=self.condition_actions,
        )
        # Drop the placeholder frame code tokens; keep the final action token.
        tokens = tokens[:, : -(self.latent_tokens)]
        generated = []
        for _ in range(self.latent_tokens):
            logits = self.forward_tokens(tokens)[:, -1, : self.codebook_size]
            logits = logits / max(float(temperature), 1e-6)
            if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
                values, indices = torch.topk(logits, k=int(top_k), dim=-1)
                probs = torch.softmax(values, dim=-1)
                next_code = indices.gather(1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(logits, dim=-1)
                next_code = torch.multinomial(probs, 1)
            generated.append(next_code)
            tokens = torch.cat([tokens, next_code], dim=1)
            if tokens.shape[1] > self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len :]
        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate_rollout(
        self,
        context_codes: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
        context_frames: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        generated = []
        codes = context_codes.long()
        actions = context_actions.long()
        for step in range(future_actions.shape[1]):
            next_code = self.generate_next_frame(
                codes[:, -context_frames:],
                actions[:, -(context_frames - 1) :] if context_frames > 1 else actions[:, :0],
                future_actions[:, step],
                temperature=temperature,
                top_k=top_k,
            )
            generated.append(next_code)
            if actions.numel() == 0:
                actions = future_actions[:, step : step + 1].long()
            else:
                actions = torch.cat([actions, future_actions[:, step : step + 1].long()], dim=1)
            codes = torch.cat([codes, next_code[:, None, :]], dim=1)
        return torch.stack(generated, dim=1)


def build_transformer_from_cfg(cfg) -> ActionConditionedTransformer:
    return ActionConditionedTransformer(
        codebook_size=int(cfg.vqvae.codebook_size),
        num_actions=int(cfg.env.num_actions),
        latent_tokens=int(cfg.transformer.latent_h) * int(cfg.transformer.latent_w),
        layers=int(cfg.transformer.layers),
        heads=int(cfg.transformer.heads),
        width=int(cfg.transformer.width),
        dropout=float(cfg.transformer.dropout),
        max_seq_len=int(cfg.transformer.max_seq_len),
        condition_actions=bool(cfg.transformer.condition_actions),
    )


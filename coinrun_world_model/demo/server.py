from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from coinrun_world_model.data.procgen import ACTION_NAMES, KEY_TO_ACTION, NOOP_ACTION
from coinrun_world_model.data.zarr_dataset import valid_sequence_starts
from coinrun_world_model.utils import encode_png_base64, resolve_device


class StepRequest(BaseModel):
    action: int = NOOP_ACTION
    temperature: float | None = None
    top_k: int | None = None


class DemoState:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = resolve_device(str(cfg.device))
        self.rng = np.random.default_rng(int(cfg.seed))
        self.vqvae = None
        self.transformer = None
        self.root = None
        self.starts = np.asarray([], dtype=np.int64)
        self.frames: list[np.ndarray] = []
        self.codes: torch.Tensor | None = None
        self.actions: torch.Tensor = torch.zeros(1, 0, dtype=torch.long, device=self.device)
        self._load_assets()

    def _load_assets(self) -> None:
        try:
            import zarr

            data_path = Path(self.cfg.data.root) / f"{self.cfg.demo.split}.zarr"
            if data_path.exists():
                self.root = zarr.open_group(str(data_path), mode="r")
                context = int(self.cfg.transformer.context_frames)
                self.starts = valid_sequence_starts(self.root, context + 1)
        except Exception:
            self.root = None

        try:
            from coinrun_world_model.evaluate import load_transformer, load_vqvae

            self.vqvae = load_vqvae(self.cfg, self.device)
            self.transformer = load_transformer(self.cfg, self.device)
        except Exception:
            if not bool(self.cfg.demo.mock_if_missing):
                raise
            self.vqvae = None
            self.transformer = None

    def reset(self) -> dict[str, Any]:
        context = int(self.cfg.transformer.context_frames)
        if self.root is not None and len(self.starts) > 0:
            start = int(self.rng.choice(self.starts))
            raw = np.asarray(self.root["frames"][start : start + context], dtype=np.uint8)
            self.frames = [frame for frame in raw]
            if self.vqvae is not None:
                tensor = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(self.device)
                with torch.no_grad():
                    self.codes = self.vqvae.encode_indices(tensor).view(1, context, -1)
            else:
                self.codes = None
            if context > 1:
                actions = np.asarray(self.root["actions"][start : start + context - 1], dtype=np.int64)
                self.actions = torch.from_numpy(actions[None]).to(self.device)
            else:
                self.actions = torch.zeros(1, 0, dtype=torch.long, device=self.device)
        else:
            self.frames = [self._mock_frame()]
            self.codes = None
            self.actions = torch.zeros(1, 0, dtype=torch.long, device=self.device)
        return self._payload()

    def step(self, action: int, temperature: float | None = None, top_k: int | None = None) -> dict[str, Any]:
        action = int(np.clip(action, 0, len(ACTION_NAMES) - 1))
        if self.vqvae is not None and self.transformer is not None and self.codes is not None:
            future_action = torch.tensor([[action]], device=self.device)
            with torch.no_grad():
                gen = self.transformer.generate_rollout(
                    self.codes,
                    self.actions,
                    future_action,
                    context_frames=min(int(self.cfg.transformer.context_frames), self.codes.shape[1]),
                    temperature=float(temperature or self.cfg.transformer.temperature),
                    top_k=int(top_k or self.cfg.transformer.top_k),
                )
                frame = self.vqvae.decode_indices(gen[:, 0]).detach()[0]
            np_frame = (
                frame.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5
            ).astype(np.uint8)
            self.frames.append(np_frame)
            self.codes = torch.cat([self.codes, gen[:, 0:1]], dim=1)
            self.actions = torch.cat([self.actions, future_action], dim=1)
            max_context = int(self.cfg.transformer.context_frames)
            self.codes = self.codes[:, -max_context:]
            self.actions = self.actions[:, -(max_context - 1) :] if max_context > 1 else self.actions[:, :0]
        else:
            self.frames.append(self._mock_next(self.frames[-1], action))
        return self._payload(action)

    def _payload(self, action: int = NOOP_ACTION) -> dict[str, Any]:
        show = int(self.cfg.demo.show_context_frames)
        return {
            "frame": encode_png_base64(self.frames[-1]),
            "context": [encode_png_base64(f) for f in self.frames[-show:]],
            "action": action,
            "action_name": ACTION_NAMES[action],
            "action_names": ACTION_NAMES,
            "key_to_action": KEY_TO_ACTION,
            "model_loaded": self.vqvae is not None and self.transformer is not None,
        }

    def _mock_frame(self) -> np.ndarray:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 35
        frame[48:56, :, :] = np.array([80, 70, 50], dtype=np.uint8)
        frame[38:48, 8:18, :] = np.array([230, 70, 80], dtype=np.uint8)
        frame[34:42, 48:56, :] = np.array([245, 210, 65], dtype=np.uint8)
        return frame

    def _mock_next(self, frame: np.ndarray, action: int) -> np.ndarray:
        shift_x = 0
        shift_y = 0
        if "RIGHT" in ACTION_NAMES[action]:
            shift_x = 2
        if "LEFT" in ACTION_NAMES[action]:
            shift_x = -2
        if "UP" in ACTION_NAMES[action]:
            shift_y = -2
        if "DOWN" in ACTION_NAMES[action]:
            shift_y = 2
        out = np.roll(frame, shift=(shift_y, shift_x), axis=(0, 1))
        out = (0.96 * out + 0.04 * self._mock_frame()).astype(np.uint8)
        return out


def create_app(cfg: Any) -> FastAPI:
    app = FastAPI(title="CoinRun World Model Demo")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = DemoState(cfg)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health():
        return {"ok": True, "model_loaded": state.vqvae is not None and state.transformer is not None}

    @app.post("/api/reset")
    def reset():
        return state.reset()

    @app.post("/api/step")
    def step(req: StepRequest):
        return state.step(req.action, req.temperature, req.top_k)

    return app


def serve_demo(cfg: Any) -> None:
    import uvicorn

    app = create_app(cfg)
    uvicorn.run(app, host=str(cfg.demo.host), port=int(cfg.demo.port))

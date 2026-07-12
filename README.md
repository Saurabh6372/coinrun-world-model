---
license: mit
tags:
- reinforcement-learning
- world-models
- procgen
- pytorch
---

# CoinRun World Model — an action-conditioned neural game engine

A world model that learns the dynamics of the [Procgen CoinRun](https://github.com/openai/procgen) environment well enough to be **played interactively**: your key presses condition an autoregressive transformer that generates the next frame — physics, rendering, and game logic all inferred by the network.

**Trained checkpoints (VQ-VAE + transformer, ~1 GB):** [MAURYASAURABH/coinrun-world-model on Hugging Face](https://huggingface.co/MAURYASAURABH/coinrun-world-model)

<!-- TODO: record a rollout with `wm demo` and add it here as docs/demo.gif -->

## How it works

Two-stage latent world model, in the spirit of world-model and neural-game-engine research:

```
64×64 RGB frame ──► VQ-VAE encoder ──► 8×8 grid of discrete tokens (codebook 512, dim 128)
                                             │
      player action (15 discrete) ──────────►│
                                             ▼
                     action-conditioned transformer (6 layers · 8 heads · width 512,
                     context 8 frames · top-k 50 / temperature 0.9 sampling)
                                             │
                                             ▼
                    next-frame tokens ──► VQ-VAE decoder ──► next 64×64 frame
```

**Data** — 250k training frames from 500 CoinRun levels, plus 25k validation and 25k test frames from *held-out level ranges* (no level leakage). Collection uses a mixed behaviour policy (50% uniform / 35% sticky / 15% platformer-biased actions) so the model sees broad action coverage, with all trajectories stored as chunked Zarr arrays.

**Training** — AMP mixed precision, AdamW (lr 3e-4, wd 0.01), config-driven via YAML with CLI overrides (see `configs/default.yaml`; `configs/mini.yaml` for a smoke-test-sized run).

## Evaluation

The eval suite (`wm evaluate` → `wm report`) measures three things that matter for interactive generation:

- **Rollout fidelity** — PSNR and SSIM at autoregressive rollout horizons 1, 2, 4, 8, 16, 32, plus an FVD-style Fréchet distance over lightweight video features.
- **Controllability** — counterfactual action tests (same context, different conditioned action) and an **inverse-dynamics probe**: a small CNN trained on real transitions, applied to generated ones, to check that the *conditioned action is actually reflected* in the generated dynamics.
- **Context sensitivity** — ablations over 1 / 4 / 8 context frames.

Metrics land in `runs/evaluate/*/metrics_summary.json` and are rendered into report + blog-post markdown by `wm report`.

<!-- TODO: paste the metrics table from outputs/report/ here once regenerated -->

## Quickstart

Python 3.10. Note: `procgen` does not build on Apple Silicon — data collection and training run on Linux/Windows + NVIDIA; the demo and tests run anywhere.

```bash
conda env create -f environment.yml
conda activate coinrun-wm
pip install -e .
```

The whole pipeline is one CLI (`wm`):

```bash
wm collect            --config configs/default.yaml   # 1. roll out CoinRun, store Zarr dataset
wm train-vqvae        --config configs/default.yaml   # 2. learn the visual tokenizer
wm encode             --config configs/default.yaml vqvae.checkpoint=runs/vqvae/latest.pt
wm train-transformer  --config configs/default.yaml   # 4. learn action-conditioned dynamics
wm evaluate           --config configs/default.yaml   # 5. rollout + controllability metrics
wm demo               --config configs/default.yaml   # 6. play it in the browser
wm report             --config configs/default.yaml   # 7. render metrics into reports
```

## Interactive demo

`wm demo` serves a local web UI (FastAPI + vanilla JS): arrow keys are mapped to CoinRun actions, each press is fed to the transformer, and the generated frames stream back — you are playing the model, not the game. A `Dockerfile` is included for containerised deployment. If no checkpoint is present the server runs in mock mode so the UI can be exercised end-to-end.

## Project structure

```
coinrun_world_model/
  data/            # procgen action space, Zarr dataset, collection policy
  models/          # VQ-VAE and action-conditioned transformer
  eval/            # PSNR/SSIM/FVD-style metrics, inverse-dynamics probe
  demo/            # FastAPI server + browser front-end
  cli.py           # `wm` entry point (typer)
  train_vqvae.py / train_transformer.py / encode.py / evaluate.py / report.py
configs/           # default.yaml, mini.yaml
tests/             # dataset, models, metrics, demo, procgen smoke tests
```

## Tests

```bash
pytest            # procgen-dependent tests skip automatically where procgen is unavailable
```

## License

MIT

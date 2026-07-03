from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from coinrun_world_model.utils import ensure_dir, read_json, write_json


def render_reports(cfg: Any) -> dict[str, Path]:
    report_dir = ensure_dir(cfg.reports.report_dir)
    blog_dir = ensure_dir(cfg.reports.blog_dir)
    metrics = _load_latest_metrics(str(cfg.reports.metrics_glob))
    plots = _render_plots(metrics, report_dir)
    report_tex = report_dir / "coinrun_world_model_report.tex"
    report_md = report_dir / "coinrun_world_model_report.md"
    blog_md = blog_dir / "action_conditioned_coinrun_world_model.md"
    report_tex.write_text(_report_tex(cfg, metrics, plots), encoding="utf-8")
    report_md.write_text(_report_md(cfg, metrics, plots), encoding="utf-8")
    blog_md.write_text(_blog_md(cfg, metrics, plots), encoding="utf-8")
    write_json(report_dir / "report_manifest.json", {"metrics": metrics, "plots": [str(p) for p in plots]})
    return {"report_tex": report_tex, "report_md": report_md, "blog_md": blog_md}


def _load_latest_metrics(pattern: str) -> dict[str, Any]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return {
            "status": "no evaluation metrics found yet",
            "rollout": {},
            "fvd_rgb_motion": None,
            "probe_generated_accuracy": None,
        }
    return read_json(matches[-1])


def _render_plots(metrics: dict[str, Any], report_dir: Path) -> list[Path]:
    rollout = metrics.get("rollout", {})
    rows = []
    for key, values in rollout.items():
        if "_step_" not in key:
            continue
        model, step = key.rsplit("_step_", 1)
        rows.append({"model": model, "step": int(step), **values})
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values(["model", "step"])
    paths = []
    for metric in ["psnr", "ssim"]:
        fig, ax = plt.subplots(figsize=(5, 3))
        for model, group in df.groupby("model"):
            ax.plot(group["step"], group[metric], marker="o", label=model)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        path = report_dir / f"{metric}_by_step.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _report_tex(cfg: Any, metrics: dict[str, Any], plots: list[Path]) -> str:
    fvd = metrics.get("fvd_rgb_motion", "TBD")
    probe = metrics.get("probe_generated_accuracy", "TBD")
    figures = "\n".join(
        rf"\includegraphics[width=0.45\textwidth]{{{path.name}}}" for path in plots
    )
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\title{{Action-Conditioned World Modeling in Procgen CoinRun}}
\author{{CoinRun World Model}}
\date{{\today}}
\begin{{document}}
\maketitle
\begin{{abstract}}
We present a compact action-conditioned video world model for Procgen CoinRun. The system learns a VQ-VAE tokenizer over 64x64 RGB frames and a causal transformer over interleaved latent frame tokens and action tokens. The project emphasizes controllable rollouts, reproducible experiment configuration, and an interactive keyboard-driven demo.
\end{{abstract}}

\section{{Introduction}}
Interactive world models must predict not only what a scene looks like next, but how interventions change that future. CoinRun is a useful small-scale setting because observations are visual, dynamics are action dependent, and the environment distribution is procedurally generated.

\section{{Dataset}}
Gameplay is collected from Procgen CoinRun with distribution mode \texttt{{{cfg.env.distribution_mode}}}. Frames are stored as RGB tensors and actions are aligned so that action $a_t$ leads from frame $x_t$ to frame $x_{{t+1}}$.

\section{{Method}}
The VQ-VAE compresses each frame to an 8x8 grid of discrete codes. The transformer consumes the stream $[z_t, a_t, z_{{t+1}}, ...]$ and predicts latent frame tokens autoregressively while treating actions as observed conditioning tokens.

\section{{Experiments}}
Rollouts are evaluated recursively from real context frames using ground-truth future actions. Metrics include PSNR and SSIM per rollout step, Fréchet video distance over deterministic RGB/motion features, and inverse-dynamics action consistency.

\section{{Results}}
Current FVD-style score: \texttt{{{fvd}}}. Generated-frame probe accuracy: \texttt{{{probe}}}.

{figures}

\section{{Ablations}}
The evaluation pipeline reports copy-last, action-conditioned, no-action, shuffled-action, and context-length variants as configured.

\section{{Limitations}}
This MVP is intentionally small. The default FVD backend is a runnable RGB/motion feature adapter rather than a canonical I3D checkpoint, and full conclusions require longer training plus repeated seeds.

\section{{Reproducibility}}
Run commands are exposed through \texttt{{wm}} and all experiments are configured by YAML with command-line overrides.
\end{{document}}
"""


def _report_md(cfg: Any, metrics: dict[str, Any], plots: list[Path]) -> str:
    plot_lines = "\n".join(f"![{p.stem}]({p.name})" for p in plots)
    return f"""# Action-Conditioned World Modeling in Procgen CoinRun

## Abstract

This report describes a VQ-VAE plus action-conditioned transformer world model for 64x64 Procgen CoinRun frames. The model predicts latent next frames under keyboard/game actions and is evaluated with rollout fidelity and controllability probes.

## Method

- VQ-VAE: 8x8 latent grid, codebook size `{cfg.vqvae.codebook_size}`, code dimension `{cfg.vqvae.code_dim}`.
- Transformer: `{cfg.transformer.layers}` layers, `{cfg.transformer.heads}` heads, width `{cfg.transformer.width}`, context `{cfg.transformer.context_frames}` frames.
- Token stream: frame latent codes followed by the action token that leads to the next frame.

## Results

- FVD-style RGB/motion score: `{metrics.get("fvd_rgb_motion", "TBD")}`
- Generated-frame probe accuracy: `{metrics.get("probe_generated_accuracy", "TBD")}`

{plot_lines}

## Limitations

The MVP prioritizes a clean, reproducible pipeline over scale. The default FVD adapter is deterministic and dependency-light; swap in a canonical pretrained video feature extractor for paper-grade comparisons.
"""


def _blog_md(cfg: Any, metrics: dict[str, Any], plots: list[Path]) -> str:
    plot_lines = "\n".join(f"![{p.stem}](../report/{p.name})" for p in plots)
    return f"""# Building a Tiny Controllable World Model for CoinRun

The goal was simple: press a key, ask the model what happens next, and make the whole experiment reproducible enough to extend.

The pipeline collects Procgen CoinRun rollouts, compresses frames with a VQ-VAE, and trains a transformer that sees both visual history and action tokens. Because the model consumes action tokens directly, the same context can be rolled forward under different counterfactual controls.

## What to Try

```bash
wm collect --config configs/mini.yaml
wm train-vqvae --config configs/mini.yaml
wm encode --config configs/mini.yaml vqvae.checkpoint=runs/vqvae/latest.pt
wm train-transformer --config configs/mini.yaml
wm evaluate --config configs/mini.yaml
wm demo --config configs/mini.yaml
```

## Early Readout

- FVD-style RGB/motion score: `{metrics.get("fvd_rgb_motion", "TBD")}`
- Generated-frame action-probe accuracy: `{metrics.get("probe_generated_accuracy", "TBD")}`

{plot_lines}

## What Failed First

The deliberately small setup will blur under long recursive rollouts. That is expected: the useful question is whether action-conditioned rollouts degrade more gracefully than copy-last and no-action ablations.

## Next Steps

Train longer, add repeated seeds, swap in canonical I3D FVD, and compare against a latent diffusion baseline once the transformer path is fully characterized.
"""


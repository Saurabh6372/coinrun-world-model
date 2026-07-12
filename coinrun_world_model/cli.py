from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from coinrun_world_model.config import load_config

app = typer.Typer(help="Action-conditioned Procgen CoinRun world-model pipeline.")
console = Console()


def _cfg(config: Path, overrides: list[str] | None):
    return load_config(config, overrides or [])


@app.command()
def collect(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    overrides: list[str] | None = typer.Argument(None),
):
    """Collect Procgen CoinRun frames/actions into Zarr splits."""
    from coinrun_world_model.data.collect import collect_dataset

    cfg = _cfg(config, overrides)
    outputs = collect_dataset(cfg)
    console.print({"outputs": {k: str(v) for k, v in outputs.items()}})


@app.command("train-vqvae")
def train_vqvae_cmd(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    overrides: list[str] | None = typer.Argument(None),
):
    """Train the frame VQ-VAE tokenizer."""
    from coinrun_world_model.train_vqvae import train_vqvae

    run_dir = train_vqvae(_cfg(config, overrides))
    console.print(f"VQ-VAE run: {run_dir}")


@app.command()
def encode(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    overrides: list[str] | None = typer.Argument(None),
):
    """Encode frames into VQ code indices."""
    from coinrun_world_model.encode import encode_dataset

    outputs = encode_dataset(_cfg(config, overrides))
    console.print({"encoded": {k: str(v) for k, v in outputs.items()}})


@app.command("train-transformer")
def train_transformer_cmd(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    overrides: list[str] | None = typer.Argument(None),
):
    """Train the action-conditioned latent transformer."""
    from coinrun_world_model.train_transformer import train_transformer

    run_dir = train_transformer(_cfg(config, overrides))
    console.print(f"Transformer run: {run_dir}")


@app.command()
def evaluate(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    overrides: list[str] | None = typer.Argument(None),
):
    """Evaluate rollout fidelity and action consistency."""
    from coinrun_world_model.evaluate import evaluate as run_evaluate

    run_dir = run_evaluate(_cfg(config, overrides))
    console.print(f"Evaluation run: {run_dir}")


@app.command()
def demo(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    overrides: list[str] | None = typer.Argument(None),
):
    """Launch the interactive keyboard demo."""
    from coinrun_world_model.demo.server import serve_demo

    cfg = _cfg(config, overrides)
    console.print(f"Demo: http://{cfg.demo.host}:{cfg.demo.port}")
    serve_demo(cfg)


@app.command()
def report(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    overrides: list[str] | None = typer.Argument(None),
):
    """Render the arXiv-style report and blog post assets."""
    from coinrun_world_model.report import render_reports

    outputs = render_reports(_cfg(config, overrides))
    console.print({"reports": {k: str(v) for k, v in outputs.items()}})


if __name__ == "__main__":
    app()


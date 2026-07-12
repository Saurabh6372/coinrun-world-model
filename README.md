# Action-Conditioned World Model for CoinRun

This repository contains a full pipeline for training an action-conditioned World Model on the Procgen CoinRun environment. The model is capable of hallucinating and simulating the game engine dynamics, physics, and rendering entirely within its neural network, conditioned on user input.

![World Model Demo](coinrun_world_model/demo/static/styles.css) <!-- Placeholder for actual demo gif if added later -->

## 🤖 Trained Models (Hugging Face)
All trained checkpoints, including the VQ-VAE and the Action-Conditioned Transformer (over 1GB of weights), are hosted publicly on Hugging Face:
**👉 [Download Models from Hugging Face (MAURYASAURABH/coinrun-world-model)](https://huggingface.co/MAURYASAURABH/coinrun-world-model) 👈**

*(Note: The live interactive web demo is fully supported locally. Hugging Face free tier currently restricts custom Docker space hosting.)*

---

## Architecture Overview

The pipeline consists of two primary models:
1. **VQ-VAE (Vector Quantized Variational Autoencoder):** Compresses the 64x64 RGB observations from the CoinRun environment into a discrete grid of latent tokens.
2. **Action-Conditioned Transformer:** An autoregressive sequence model that takes the history of latent tokens and the player's chosen actions to predict the next future state of the game.

## Installation

The environment requires Python 3.10. *(Note: Data collection requires `procgen`, which is incompatible with Apple Silicon Macs. Training is fully supported on Windows/Linux with NVIDIA GPUs.)*

```bash
conda env create -f environment.yml
conda activate coinrun-wm
pip install -e .
```

## Pipeline Usage

The project provides a unified CLI via the `wm` command. See `configs/default.yaml` for hyperparameters.

### 1. Data Collection
Generate the dataset by simulating the environment with various agent policies.
```bash
wm collect --config configs/default.yaml
```

### 2. VQ-VAE Training
Train the visual tokenizer to compress frames into discrete latent codes.
```bash
wm train-vqvae --config configs/default.yaml
```

### 3. Latent Encoding
Pre-process the dataset by running all frames through the trained VQ-VAE.
```bash
wm encode --config configs/default.yaml vqvae.checkpoint=runs/vqvae/latest.pt
```

### 4. Transformer Training
Train the causal sequence model on the encoded latent trajectories and actions.
```bash
wm train-transformer --config configs/default.yaml
```

### 5. Local Interactive Demo
Launch the interactive web UI to play the world model locally.
```bash
wm demo --config configs/default.yaml
```

## Deployment
A `Dockerfile` is included in the repository for easy deployment of the interactive demo to Hugging Face Spaces or any Docker-compatible hosting environment.

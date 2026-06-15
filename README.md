# CoinRun World Model

Hey everyone! 👋 This is my personal side project that I started back in mid-June to dive deeper into reinforcement learning and world models. 

The goal here is pretty straightforward but challenging: I wanted to build a world model that can learn and simulate the dynamics of the Procgen CoinRun environment, conditioned on the agent's actions. 

I've been chipping away at this over the last few weeks, working on different components of the pipeline, from data collection to training the models and finally putting together an interactive demo to see how well it learned.

## Project Timeline 📅

Here's a quick look at how the project evolved since I started:

* **June 15:** Kicked off the project. Set up the basic environment, dependencies, and repo structure.
* **June 18:** Started drafting the core architectures. Added the foundational models and configurations to handle the 64x64 RGB observations.
* **June 22:** Worked on the data pipeline. Wrote scripts to collect rollouts from CoinRun to build up a solid dataset for training.
* **June 26:** Implemented and trained the VQ-VAE. This was crucial for compressing the visual frames into discrete latent codes to make sequence modeling tractable.
* **June 30:** Brought in the Transformer! Wrote the causal sequence modeling code to predict future states based on the current state and action.
* **July 3:** Built out the evaluation metrics. Needed a way to objectively measure rollout quality, so I added some probing and distance metrics.
* **July 6:** Put the finishing touches on the codebase. Wrapped everything up into a clean CLI and built a neat little web demo so I can manually control the agent in the "dream" environment!
* **July 7:** Final cleanups, bug fixes, and getting everything ready to push.

## How to use it

I'm using Procgen 0.10.7 which works well on Python 3.7-3.10.

```bash
conda env create -f environment.yml
conda activate coinrun-wm
```

There are a few main commands you can run through the CLI (checkout `configs/default.yaml` for my training params):

1. **Collect Data:** Grab frames and actions from the environment.
   ```bash
   wm collect --config configs/default.yaml
   ```
2. **Train the Tokenizer:** Train the VQ-VAE on the collected frames.
   ```bash
   wm train-vqvae --config configs/default.yaml
   ```
3. **Encode Data:** Convert frames to latent codes using the trained VQ-VAE.
   ```bash
   wm encode --config configs/default.yaml vqvae.checkpoint=runs/vqvae/latest.pt
   ```
4. **Train the World Model:** Train the causal transformer on the encoded sequences.
   ```bash
   wm train-transformer --config configs/default.yaml
   ```
5. **Play:** Launch the interactive demo to test the model!
   ```bash
   wm demo --config configs/default.yaml
   ```

## Notes

- CoinRun observations are 64x64 RGB and the action space is discrete (15 possible actions).
- I optimized this to run smoothly on a single GPU for full training, though I did some testing on CPU/Apple MPS for the mini runs.
- I'm really proud of how the evaluation pipeline turned out, especially the Fréchet distance setup which runs without needing massive pretrained backends like I3D.

Feel free to poke around the code or run the demo yourself! Let me know if you have any questions or suggestions.

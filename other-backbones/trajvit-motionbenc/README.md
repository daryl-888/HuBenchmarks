# TrajViT × MotionBench — NOT YET RUNNABLE

**Status: VideoQA evaluation not yet supported by TrajViT.**

TrajViT (ICCV 2025 Highlight) — "One Trajectory, One Token: Grounded Video Tokenization via Panoptic Sub-object Trajectory"

GitHub: [RAIVNLab/trajvit](https://github.com/RAIVNLab/trajvit)

---

## What is TrajViT?

TrajViT is a video encoder that replaces fixed space-time patch tokens with trajectory tokens built from panoptic sub-object trajectories. Each moving object/sub-object through the video becomes one compact token.

- **Architecture:** Trained trajectory-aware video encoder (not plug-and-play)
- **Evaluation scope (current):** Zero-shot video-text retrieval only
- **README note:** *"We will update the repo soon with evaluation scripts on more tasks"*

---

## Why this folder exists

TrajViT is architecturally the most distinct method in this benchmark set — it re-tokenizes the video around trajectories rather than pruning standard patch tokens. When VideoQA eval lands, MotionBench is a natural target: trajectory tokens are explicitly motion-aware.

---

## What to do when VideoQA eval is released

1. Check [RAIVNLab/trajvit](https://github.com/RAIVNLab/trajvit) for new evaluation scripts
2. Confirm whether TrajViT provides an lmms_eval backend or its own eval framework
3. If lmms_eval: add `run_trajvit.sbatch` using `--model trajvit`
4. If custom: write `eval_motionbench.py` following the STTM/PruneVid pattern in this repo

---

## Do not run yet

The sbatch files here are removed until TrajViT releases VideoQA support.
No fake commands are provided.

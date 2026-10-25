# HuBenchmarks — Project Brief

Evaluating training-free video-LLM efficiency methods on the **MotionBench**
benchmark at UH Carya HPC.

**Scope:** 11 methods × 2 backbones (LLaVA-OV-7B, Qwen3-VL-8B), standardized to
32 frames and 15% retention where the method exposes one. Directories are
organized by backbone generation (`stage1-llava-ov/`, `stage3-qwen3-vl/`,
`other-backbones/`), one subdirectory per method.

## Core Mission

Benchmark efficient video understanding models on MotionBench (8,052 samples, ~4,018 scoreable, 5,385 unique videos) and record accuracy results. The cluster environment has strict permission constraints that require workarounds for env setup.

## Key Constraints

- **No inference on login node** — all GPU jobs through `sbatch`
- **Batch size must be 1** for video models
- **Conda env permission wall**: many envs owned by `mahern69` (not `dpalfaro`), so `pip install --target` is the standard workaround for missing packages
- **Stale sbatch syndrome**: files must be synced to Carya via `./scripts/deploy.sh --push <subtree>` after every local edit — git push alone does NOT update Carya
- **NFS stale video handles**: `load_video` must use subprocess isolation with hard kill timeout (see CLAUDE.md)
- **JSONL data quirks**: `video_info.meta.jsonl` has mixed-type `resolution` field (both array and object formats) — pyarrow >= 24.0.0 requires normalization
- **A number is not a result until it is gated**: a run must log that the method
  engaged AND show predictions differing from the plain backbone. Three ports
  completed cleanly while doing nothing; only the divergence check caught them.

# HuBenchmarks — Project Brief

Evaluating video LLMs on the **MotionBench** benchmark at UH Carya HPC. Each subdirectory in this repo corresponds to one model.

## Core Mission

Benchmark efficient video understanding models on MotionBench (8,052 samples, ~4,018 scoreable, 5,385 unique videos) and record accuracy results. The cluster environment has strict permission constraints that require workarounds for env setup.

## Key Constraints

- **No inference on login node** — all GPU jobs through `sbatch`
- **Batch size must be 1** for video models
- **Conda env permission wall**: many envs owned by `mahern69` (not `dpalfaro`), so `pip install --target` is the standard workaround for missing packages
- **Stale sbatch syndrome**: files must be synced to Carya via heredoc/scp after every local edit — git push alone does NOT update Carya
- **NFS stale video handles**: `load_video` must use subprocess isolation with hard kill timeout (see CLAUDE.md)
- **JSONL data quirks**: `video_info.meta.jsonl` has mixed-type `resolution` field (both array and object formats) — pyarrow >= 24.0.0 requires normalization

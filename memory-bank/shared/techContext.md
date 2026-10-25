# Tech Context — HuBenchmarks

## Technologies Used

- **HPC cluster**: UH Carya (SLURM scheduler, Ada GPUs)
- **Python**: 3.10–3.11 across conda envs
- **PyTorch**: Multiple versions — 2.2.0 (DyTo), 2.3.1 (MDP3/AIM via --target), 2.6.0 (broken in mdp3 env)
- **Backbone LLMs**: LLaVA-OV-7B (Stage 1, `LlavaQwenForCausalLM` — Qwen2
  internally; there is no separate "LLaVA-OV-Qwen2" model, that was a
  misconception and the duplicate weights were deleted), Qwen3-VL-8B (Stage 3,
  transformers 5.14.1), LLaVA-NeXT Vicuna-7B (DyTo), PLLaVA-7B (PruneVid),
  LLaVA-1.5-7B (VisionZip)
- **lmms_eval**: Modified fork for video evaluation
- **datasets/pyarrow**: huggingface datasets (envs use either 2.16.1+pyarrow24 or 4.8.5+pyarrow24)

## Cluster Paths

| Path | Purpose |
|------|---------|
| `/project/rhu/dpalfaro/conda/envs/` | All conda envs |
| `/project/rhu/dpalfaro/code/` | Synced from this repo (mostly mahern69-owned) |
| `/project/rhu/dpalfaro/DYTO` | DyTo source (dpalfaro-owned) |
| `/project/rhu/dpalfaro/AIM` | AIM source (dpalfaro-owned) |
| `/project/rhu/dpalfaro/mdp3_pkgs/` | torch 2.3.1+cu121 (--target) |
| `/project/rhu/dpalfaro/aim_pkgs/` | torch 2.3.1+cu121 (--target) |
| `/project/rhu/dpalfaro/dyto_pkgs/` | finch-clust 0.2.0 (--target) |
| `/project/rhu/MotionBench_Data/MotionBench/` | Videos + JSONL |
| `/project/rhu/dpalfaro/cache/huggingface/` | HF cache (models + datasets) |
| `/project/rhu/dpalfaro/results/` | Job output |

## Conda Env Setup Summary

| Env | Owner | Base | Notes |
|-----|-------|------|-------|
| dycoke11 | dpalfaro | — | Primary env, working |
| sttm | dpalfaro | clone dycoke11 | Working |
| prunevid | dpalfaro | — | Working |
| holitom | dpalfaro | — | transformers==4.45.2 |
| videoitg | dpalfaro | — | Working |
| flashvid | dpalfaro | clone dycoke11 | Working |
| mdp3 | mahern69 | clone holitom | **Broken torch**; fix: --target mdp3_pkgs |
| aim | mahern69 | — | **Broken torch/torchvision**; fix: --target aim_pkgs |
| dyto | mahern69 | — | **Missing finch-clust**; fix: --target dyto_pkgs |

## Permission Model

- `/project/rhu/dpalfaro/` root is dpalfaro-owned (writable)
- `/project/rhu/dpalfaro/code/*` — some subdirs are mahern69-owned (EACCES on write)
- `/project/rhu/dpalfaro/conda/envs/` — individual envs may be mahern69-owned
- `/project/rhu/MotionBench_Data/` — dpalfaro owns the JSONL (fixable), videos are read-only

## Carya Access

Works via SSH with key:
```bash
ssh dpalfaro@carya.rcdc.uh.edu
```
No password prompt — key-based auth. SCP for file transfer.

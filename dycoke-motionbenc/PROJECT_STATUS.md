---
name: project-status
description: Current status of DyCoke x MotionBench evaluation project on UH Carya cluster
metadata: 
  node_type: memory
  type: project
  originSessionId: 45a65144-ea64-4128-a7fb-d1f4a72aeec7
---

# DyCoke x MotionBench — Project Status

## What this project is
Evaluating DyCoke (training-free KV cache compression for video LLMs) on MotionBench benchmark using LLaVA-OV-7B on UH Carya HPC cluster. Extending DyCoke's original evaluation to a motion-specific benchmark not in the paper.

## Repo
- Local: `/home/bung/Projects/HuVLLM/`
- GitHub: `git@github.com:daryl-888/HuBenchmarks.git`
- Post-commit hook set up to auto-push on commit

## Cluster
- Carya HPC at UH: `ssh -l dpalfaro carya.rcdc.uh.edu`
- User: `dpalfaro`
- Project dir on Carya: `/project/rhu/dpalfaro/`
- Code dir: `/project/rhu/dpalfaro/code/dycoke-motionbenc/`
- Files transferred via `scp` from local to Carya (not git pull)
- Ada GPU nodes work (`compute-9-[1-8]`, `compute-10-[1-12]`) — Volta nodes have old drivers, avoid
- Use `--gpus=ada:1` in sbatch to request any Ada node

## Key paths on Carya
- Weights: `/project/rhu/dpalfaro/weights/llava-ov-7b`
- Dataset: `/project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl`
- Cache: `/project/rhu/dpalfaro/cache/huggingface/`
- Results: `/project/rhu/dpalfaro/results/`
- Conda env: `/project/rhu/dpalfaro/conda/envs/dycoke11/`
- DyCoke source: `/project/rhu/dpalfaro/code/DyCoke/`

## Current run status (as of 2026-05-26)
- DyCoke full run: job 6972483 on compute-10-7 — RUNNING
- Baseline full run: job 6972484 on compute-10-9 — RUNNING
- Both using 12 hour time limit, Ada GPU, 8 cores, 32GB RAM
- Results go to `dycoke_run1/` and `baseline_run1/` respectively

## Key fixes made
1. Removed `HF_DATASETS_OFFLINE=1` from sbatch — was blocking json dataset loader
2. Added `unset HF_HUB_OFFLINE` and `unset HF_DATASETS_OFFLINE` to all sbatch files
3. Kept `dataset_path: json` in yaml (NOT a local path — lmms_eval treats local paths as HF repo IDs)
4. Removed `video: True` from yaml dataset_kwargs — was triggering HF Hub snapshot_download
5. NA samples (4034/8052 = 50%) skipped in scoring per official MotionBench protocol
6. DyCoke params: `l=3, p=0.8, k=0.3` (matches repo defaults in builder.py)

## Dataset notes
- 8052 total samples, 4034 have `answer: NA` (unanswerable) — skip per official eval
- Only 4018 samples are scoreable
- Dataset cache already built at `/project/rhu/dpalfaro/cache/huggingface/datasets/`

## Files in repo
- `run_dycoke.sbatch` — full DyCoke evaluation run
- `run_baseline.sbatch` — baseline run (dycoke=False)
- `test_dycoke.sbatch` — 10 sample sanity check (--limit 10)
- `cache_dataset.sbatch` — one-time dataset cache build (already done, don't need to rerun)
- `tasks/motionbench/motionbench.yaml` — lmms_eval task config
- `tasks/motionbench/utils.py` — dataset loading and scoring functions
- `README.md` — full project documentation
- `CONTRIBUTIONS.md` — research context and what results will show
- `TEST_RESULTS.md` — test run results (10 samples, 40% accuracy on 5 answerable)

## Known issues / watch out for
- `HF_HUB_OFFLINE` may be set system-wide on some Carya nodes — always `unset` it in sbatch
- Progress bar only shows in `.err` file, not `.out` file
- `.out` file only gets final results appended after full run completes
- Git divergence: remote has auto-sync commits — use `git pull --rebase origin main` to sync

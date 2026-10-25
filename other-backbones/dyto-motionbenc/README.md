# DyTo × MotionBench

**DyTo (Dynamic Token Merging)** — ICCV 2025 — is a training-free zero-shot video understanding method that optimizes token efficiency through hierarchical frame selection and bipartite token merging.

**Paper:** [Beyond Training: Dynamic Token Merging for Zero-Shot Video Understanding](https://arxiv.org/abs/2411.14401)

**Source:** https://github.com/Jam1ezhang/DYTO

## Backbone

LLaVA-NeXT-Vicuna-7B (NOT LLaVA-OV-Qwen — different architecture).

| Property | Value |
|----------|-------|
| Weights | `/project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b` |
| Conv template | `vicuna_v1` |
| Source | `liuhaotian/llava-v1.6-vicuna-7b` on HuggingFace |

## Setup on Carya

```bash
# 1. Clone DYTO
cd /project/rhu/dpalfaro/code
git clone https://github.com/Jam1ezhang/DYTO

# 2. Create conda env (clone from dycoke11 for compatible deps)
conda create --name dyto --clone dycoke11
/project/rhu/dpalfaro/conda/envs/dyto/bin/pip install -e /project/rhu/dpalfaro/code/DYTO

# 3. Download Vicuna-7B weights (login node, has internet)
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b \
    /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b

# 4. Create dyto-motionbenc dir on Carya
ssh dpalfaro@carya.rcdc.uh.edu "mkdir -p /project/rhu/dpalfaro/code/dyto-motionbenc/tasks/motionbench"

# 5. SCP files from local repo
scp dyto-motionbenc/eval_dyto.py          dpalfaro@carya:/project/rhu/dpalfaro/code/dyto-motionbenc/
scp dyto-motionbenc/run_dyto.sbatch       dpalfaro@carya:/project/rhu/dpalfaro/code/dyto-motionbenc/
scp dyto-motionbenc/test_dyto.sbatch      dpalfaro@carya:/project/rhu/dpalfaro/code/dyto-motionbenc/
scp dyto-motionbenc/tasks/motionbench/*   dpalfaro@carya:/project/rhu/dpalfaro/code/dyto-motionbenc/tasks/motionbench/
```

## Running

### Smoke test (50 samples, ~1 hour)
```bash
sbatch /project/rhu/dpalfaro/code/dyto-motionbenc/test_dyto.sbatch
```

### Full evaluation (8,052 samples, ~24 hours)
```bash
sbatch /project/rhu/dpalfaro/code/dyto-motionbenc/run_dyto.sbatch
```

## Results

Results are written to `/project/rhu/dpalfaro/results/dyto_run1/`:
- `results.jsonl` — per-sample predictions
- `summary.json` — accuracy summary

## Notes

- DyTo's token merging is controlled by the `TEMPORAL_AGGREGATION` env var.
  Set to `cluster` to enable DyTo, or unset/empty for baseline LLaVA-NeXT.
- The NFS stale-handle fix (subprocess isolation with 60s timeout) is built into `eval_dyto.py`.
- This is a **native pipeline** (not lmms_eval) — follows the same pattern as HoliTom and MDP3.
- Results are **not directly comparable** to Qwen-based models (DyCoke, HoliTom, VideoITG, etc.)
  due to the different backbone architecture.
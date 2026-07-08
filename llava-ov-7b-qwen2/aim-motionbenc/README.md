# AIM × MotionBench

**AIM (Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning)** — ICCV 2025 — is a training-free method that reduces visual token redundancy through:
1. **Token Merging** (before LLM): Iteratively merges visually similar tokens based on cosine similarity
2. **Token Pruning** (inside LLM layers): Progressively prunes unimportant visual tokens using PageRank on attention weights

**Paper:** [AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](https://arxiv.org/abs/2412.03248)

**Source:** https://github.com/LaVi-Lab/AIM

## Backbone

LLaVA-OneVision-7B (Qwen2 LLM — same as FlashVID).

| Property | Value |
|----------|-------|
| Weights | `/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2` |
| Conv template | `qwen_1_5` |
| Source | `lmms-lab/llava-onevision-qwen2-7b-ov` on HuggingFace |
| LLM | Qwen2 (28 layers) |

## Setup on Carya

```bash
# 1. Clone AIM
cd /project/rhu/dpalfaro/code
git clone https://github.com/LaVi-Lab/AIM

# 2. Create conda env (clone from dycoke11 for compatible deps)
conda create --name aim --clone dycoke11

# 3. Install AIM
/project/rhu/dpalfaro/conda/envs/aim/bin/pip install -e /project/rhu/dpalfaro/code/AIM

# 4. Install AIM's custom packages (transformers fork, lmms-eval fork, qwen-vl-utils)
cd /project/rhu/dpalfaro/code/AIM/other_packages/transformers && \
  /project/rhu/dpalfaro/conda/envs/aim/bin/pip install -e .

cd /project/rhu/dpalfaro/code/AIM/other_packages/lmms-eval && \
  /project/rhu/dpalfaro/conda/envs/aim/bin/pip install -e .

cd /project/rhu/dpalfaro/code/AIM/other_packages/qwen-vl-utils && \
  /project/rhu/dpalfaro/conda/envs/aim/bin/pip install -e .

# 5. Create aim-motionbenc dir and SCP files
ssh dpalfaro@carya.rcdc.uh.edu "mkdir -p /project/rhu/dpalfaro/code/aim-motionbenc/tasks/motionbench"
```

## Running

### Smoke test (50 samples, ~1 hour)
```bash
sbatch /project/rhu/dpalfaro/code/aim-motionbenc/test_aim.sbatch
```

### Full evaluation (8,052 samples, ~48 hours)
```bash
sbatch /project/rhu/dpalfaro/code/aim-motionbenc/run_aim.sbatch
```

## Configuration

AIM is configured via model_args in the sbatch file. Key params:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attn_implementation=eager` | Required | Eager attention — needed for token pruning (reads weights) |
| Merge ratio | 12.5% (hardcoded in `llava_arch.py`) | 3 merge passes: 50% → 25% → 12.5% |
| `l1` | 14 | Pruning start layer (Table 5, Exp 6) |
| `l2` | 22 | Pruning end layer (Table 5, Exp 6) |

To adjust the merge ratio, edit `llava_arch.py` line 195 (`orig_num//2` for 50%, `orig_num//4` for 25%, etc.).

## Results

Results are written to `/project/rhu/dpalfaro/results/aim_run1/`:
- `*_samples_motionbench.jsonl` — per-sample predictions
- `motionbench.json` — aggregate results

## Notes

- This uses lmms_eval (same as DyCoke/STTM) via accelerate launch.
- `attn_implementation=eager` is required for token pruning — don't use `sdpa`.
- AIM shares the same Qwen2 backbone as FlashVID — results are directly comparable.
- Weights are already on Carya at `/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2`.
# HuVLLM Documentation

Efficient video-LLM methods evaluated on **MotionBench**, across two backbones.
This folder is the replication hub: start here, then follow the links.

## Contents

| Page | What |
|---|---|
| [SETUP.md](SETUP.md) | Weights, datasets, conda environments, and the `config/paths.sh` you must edit to run anywhere. |
| [RESULTS.md](RESULTS.md) | The authoritative results table — every method × backbone, with subcategories and verification status. |
| [METHODOLOGY.md](METHODOLOGY.md) | How a run is defined, scored, and — critically — **verified to actually engage** (the anti-silent-failure gate). |
| [DETERMINISM_AND_VALIDITY.md](DETERMINISM_AND_VALIDITY.md) | Why re-runs reproduce exactly, what that proves — and where this methodology is **flawed** (incl. significance testing). |
| [VALIDITY_ASSESSMENT.md](VALIDITY_ASSESSMENT.md) | Critical self-audit: honesty & scoring-accuracy ratings per layer, with the significance tests behind them. |
| [backbones/](backbones/) | One page per backbone: architecture, which methods run on it, and per-category baseline. |
| [methods/](methods/) | One page per method: the algorithm, our port, parameters, and how to reproduce it. |

## The benchmark in one paragraph

MotionBench is 8,052 multiple-choice video questions (4,018 scoreable after
removing 4,034 `NA` items). We evaluate **training-free efficiency methods**
(token pruning / merging / frame selection) on two backbones —
**LLaVA-OV-7B** (mid-2024) and **Qwen3-VL-8B** (late-2025) — at a standardized
**32 frames** and **15% visual-token retention** where the method exposes such a
knob. Decoding is greedy (`do_sample=False`), answers are letter-matched.

## Headline result

**The backbone dominates the method.** Qwen3-VL-8B baseline scores **62.52%** vs
LLaVA-OV-7B's **52.66%** — a ~10-point gap that no efficiency method on either
backbone comes close to. On LLaVA-OV, every method lands within ~0.8 points of the
backbone (i.e. within noise). See [RESULTS.md](RESULTS.md).

## Quick start

```bash
# 1. point the config at your weights / dataset / results
cp config/paths.sh config/paths.local.sh   # edit this copy
source config/paths.local.sh

# 2. run one method (no SLURM needed — this is the underlying command)
python stage1-llava-ov/dycoke-motionbenc/eval_dycoke.py \
    --model_path "$W_LLAVA_OV" \
    --meta_path  "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/dycoke_run" \
    --num_frames 32 --conv_template qwen_2

# 3. verify it actually engaged (did not silently degrade to the backbone)
python scripts/check_run.py "$HUVLLM_RESULTS/dycoke_run" \
    --expect-method dycoke --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The `.sbatch` files beside each `eval_*.py` are our exact Carya submissions, kept
as reference. `docs/methods/<method>.md` gives the per-method command.

# DyCoke

**Paper** arXiv 2411.14401 · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

Two stages: (K) temporal token **merging** across frames, then (P) dynamic **KV-cache pruning** at LLM layer l during generation.

## Our implementation

- **Params (standardized):** `l=3, p=0.7, k=0.7`
- **ACTIVE log** (confirms engagement in stderr): `(uses DyCoke's patched PrunableDynamicCache; see stderr)`
- Source: `stage1-llava-ov/dycoke-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/dycoke-motionbenc/eval_dycoke_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 53.36% (2144) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 53.36% (2144) | 38.9 / 48.8 / 55.7 / 58.2 / 70.9 / 25.2 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/dycoke-motionbenc/eval_dycoke.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/dycoke_run" --num_frames 32 --conv_template qwen_2
python scripts/check_run.py "$HUVLLM_RESULTS/dycoke_run" \
    --expect-method dycoke --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/dycoke-motionbenc/w2_run_dycoke.sbatch`.

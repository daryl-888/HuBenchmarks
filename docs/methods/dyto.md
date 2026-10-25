# DyTo

**Paper** ICCV 2025 · **Native backbone** LLaVA-NeXT Vicuna-7B · **Category** token/frame reduction, training-free

## Algorithm

Training-free **dynamic token merging**: FINCH clustering picks ~25 representative frames from 100, ToMe merges tokens per-frame to a dynamic budget.

## Our implementation

- **Params (standardized):** `temporal_aggregation=spatial_tome_finch_dynamic_all_frms, rope_scaling=2`
- **ACTIVE log** (confirms engagement in stderr): `(triggered by temporal_aggregation kwarg)`
- Source: `stage1-llava-ov/dyto-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/dyto-motionbenc/eval_dyto_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — pending ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| pending | — |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/dyto-motionbenc/eval_dyto.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/dyto_run" --num_frames 32 --conv-template vicuna_v1 --num-frames 100
python scripts/check_run.py "$HUVLLM_RESULTS/dyto_run" \
    --expect-method dyto --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/dyto-motionbenc/w2_run_dyto.sbatch`.

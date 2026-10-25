# STTM

**Paper** 2025 · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

**Quadtree** spatio-temporal token merging patched into Qwen2 attention (`replace_qwen2_with_quadtree_attn`). Efficiency comes from LLM layers, vision runs normally.

## Our implementation

- **Params (standardized):** `sa_start_layer_idx=2, sa_tree_thresh=0.85`
- **ACTIVE log** (confirms engagement in stderr): `(quadtree attn patch)`
- Source: `stage1-llava-ov/sttm-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/sttm-motionbenc/eval_sttm_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 51.72% (2078) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 51.72% (2078) | 39.9 / 48.1 / 53.8 / 53.5 / 70.6 / 28.8 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/sttm-motionbenc/eval_sttm.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/sttm_run" --num_frames 32 --sa_start_layer_idx 2 --sa_tree_thresh 0.85
python scripts/check_run.py "$HUVLLM_RESULTS/sttm_run" \
    --expect-method sttm --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/sttm-motionbenc/w2_run_sttm.sbatch`.

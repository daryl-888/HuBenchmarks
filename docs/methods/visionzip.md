# VisionZip

**Paper** 2024 · **Native backbone** LLaVA-1.5-7B · **Category** token/frame reduction, training-free

## Algorithm

Selects **dominant tokens** (highest CLS-attention) + **contextual tokens** (spatial neighbours) inside the CLIP vision tower.

## Our implementation

- **Params (standardized):** `dominant=54, contextual=10`
- **ACTIVE log** (confirms engagement in stderr): `(vision-tower token selection)`
- Source: `stage1-llava-ov/visionzip-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/visionzip-motionbenc/eval_visionzip_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 39.97% (other-backbones) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 39.97% (other-backbones) | — (see other-backbones) |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/visionzip-motionbenc/eval_visionzip.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/visionzip_run" --num_frames 32 --model_name llava_v1.5_7b --conv_template llava_v1
python scripts/check_run.py "$HUVLLM_RESULTS/visionzip_run" \
    --expect-method visionzip --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/visionzip-motionbenc/w2_run_visionzip.sbatch`.

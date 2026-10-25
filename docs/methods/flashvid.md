# FlashVID

**Paper** ICLR 2026 (Oral) · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

Pre-LLM temporal-segment token **merge**: segment the video, then keep a retention budget per segment scored by saliency + temporal distinctiveness.

## Our implementation

- **Params (standardized):** `retention_ratio=0.15, alpha=0.7`
- **ACTIVE log** (confirms engagement in stderr): `(FlashVID wraps the model at load)`
- Source: `stage1-llava-ov/flashvid-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/flashvid-motionbenc/eval_flashvid_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 53.31% (2142) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 53.31% (2142) | 39.9 / 45.7 / 53.8 / 58.0 / 71.7 / 28.2 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/flashvid-motionbenc/eval_flashvid.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/flashvid_run" --num_frames 32 --retention_ratio 0.15 --alpha 0.7
python scripts/check_run.py "$HUVLLM_RESULTS/flashvid_run" \
    --expect-method flashvid --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/flashvid-motionbenc/w2_run_flashvid.sbatch`.

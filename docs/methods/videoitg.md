# VideoITG

**Paper** 2025 · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

Two-stage **grounded frame selection**: a grounding pass scores frames per question (`frame_scores.jsonl`), an inference pass consumes the selected `frame_indices`.

## Our implementation

- **Params (standardized):** `grounding: 512 sample, 32 select, 2fps`
- **ACTIVE log** (confirms engagement in stderr): `(two-stage: grounding -> infer; needs frame_scores.jsonl)`
- Source: `stage1-llava-ov/videoitg-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/videoitg-motionbenc/eval_videoitg_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 52.86% (2124) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 52.86% (2124) | 40.1 / 47.0 / 53.7 / 57.6 / 70.1 / 26.8 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/videoitg-motionbenc/eval_videoitg.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/videoitg_run" --num_frames 32 --grounding_jsonl <frame_scores.jsonl>
python scripts/check_run.py "$HUVLLM_RESULTS/videoitg_run" \
    --expect-method videoitg --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/videoitg-motionbenc/w2_run_videoitg.sbatch`.

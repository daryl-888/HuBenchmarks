# HoliTom

**Paper** 2025 · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

Holistic merging in two stages: **outer** (pre-LLM) temporal-segment retain, **inner** (in-LLM) attention merge from layer k.

## Our implementation

- **Params (standardized):** `RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5`
- **ACTIVE log** (confirms engagement in stderr): `(env-var driven: WRAPPER, RETAIN_RATIO, T, HOLITOM_k, HOLITOM_r)`
- Source: `stage1-llava-ov/holitom-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/holitom-motionbenc/eval_holitom_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 53.14% (2135) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 53.14% (2135) | 40.8 / 49.6 / 52.6 / 57.0 / 71.4 / 27.2 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/holitom-motionbenc/eval_holitom.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/holitom_run" --num_frames 32 env RETAIN_RATIO=0.15 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5
python scripts/check_run.py "$HUVLLM_RESULTS/holitom_run" \
    --expect-method holitom --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/holitom-motionbenc/w2_run_holitom.sbatch`.

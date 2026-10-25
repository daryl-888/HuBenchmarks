# AIM

**Paper** ICCV 2025 · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

**Bipartite soft-matching** token merge (50→25→12.5→6.25%) then **PageRank** pruning to keep the most central tokens. Compiled into a patched llava_arch.py.

## Our implementation

- **Params (standardized):** `bipartite merge (4 steps) + PageRank`
- **ACTIVE log** (confirms engagement in stderr): `(compiled into llava_arch.py)`
- Source: `stage1-llava-ov/aim-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/aim-motionbenc/eval_aim_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 52.86% (2124) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 52.86% (2124) | 41.4 / 48.1 / 54.2 / 57.0 / 71.9 / 22.2 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/aim-motionbenc/eval_aim.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/aim_run" --num_frames 32 
python scripts/check_run.py "$HUVLLM_RESULTS/aim_run" \
    --expect-method aim --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/aim-motionbenc/w2_run_aim.sbatch`.

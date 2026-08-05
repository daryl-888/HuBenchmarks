# FastV

**Paper** arXiv 2403.06764 · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

**Attention-rerank pruning**: at layer K, rank image tokens by the attention they receive (last query row), keep the top fraction, mask the rest for all layers ≥ K.

## Our implementation

- **Params (standardized):** `k=2, r=0.85 (keeps 15%)`
- **ACTIVE log** (confirms engagement in stderr): `FastV ACTIVE: img_len=6273 keep=941 (dropped 5332) seq=6344`
- Source: `stage1-llava-ov/fastv-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/fastv-motionbenc/eval_fastv_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 36.78% (1478) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 36.78% (1478) | 32.8 / 31.9 / 34.1 / 35.7 / 53.8 / 25.2 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/fastv-motionbenc/eval_fastv.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/fastv_run" --num_frames 32 --fastv --fastv_k 2 --fastv_r 0.85
python scripts/check_run.py "$HUVLLM_RESULTS/fastv_run" \
    --expect-method fastv --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/fastv-motionbenc/w2_run_fastv.sbatch`.

# PruneVID

**Paper** 2024 · **Native backbone** PLLaVA-7B · **Category** token/frame reduction, training-free

## Algorithm

**Video Token Pruning** (VTP): cluster spatially-temporally redundant tokens and prune. Bound to PLLaVA — its LLaVA-OV port was inert (0/8052 divergence).

## Our implementation

- **Params (standardized):** `cluster_ratio=0.5, temporal_segment_ratio=0.25`
- **ACTIVE log** (confirms engagement in stderr): `(pruning_enabled: True in summary)`
- Source: `stage1-llava-ov/prunevid-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/prunevid-motionbenc/eval_prunevid_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 44.13% (other-backbones) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 44.13% (other-backbones) | 35.6 / 33.0 / 41.0 / 47.0 / 63.3 / 26.5 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/prunevid-motionbenc/eval_prunevid.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/prunevid_run" --num_frames 32 use_lora=True (PLLaVA PEFT)
python scripts/check_run.py "$HUVLLM_RESULTS/prunevid_run" \
    --expect-method prunevid --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/prunevid-motionbenc/w2_run_prunevid.sbatch`.

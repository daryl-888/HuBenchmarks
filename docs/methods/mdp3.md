# MDP3

**Paper** ICCV 2025 · **Native backbone** LLaVA-OV-7B · **Category** token/frame reduction, training-free

## Algorithm

Conditional **determinantal point process** frame selection: pick the most informative, least-redundant frames from a pool, conditioned on the question. Acts BEFORE the model.

## Our implementation

- **Params (standardized):** `pool_frames=32, select_frames=8`
- **ACTIVE log** (confirms engagement in stderr): `MDP3(...) ACTIVE: pool=N -> selected=M frames`
- Source: `stage1-llava-ov/mdp3-motionbenc/` (LLaVA-OV) and
  `stage3-qwen3-vl/mdp3-motionbenc/eval_mdp3_qwen3vl.py` (Qwen3-VL port, if applicable).

## Results

### LLaVA-OV-7B — 53.06% (2132) ✅ gated

| Overall | Act.Order / Cam.Motion / Loc.Motion / Mot.Rec. / Mot.Obj. / Rep.Count |
|:---:|---|
| 53.06% (2132) | 40.5 / 49.6 / 53.5 / 56.8 / 71.6 / 26.5 |

*(backbone baseline: 52.66%)*

### Qwen3-VL-8B

See [backbones/qwen3-vl-8b.md](../backbones/qwen3-vl-8b.md) for port status.
Baseline on this backbone: **62.52%**.

## Reproduce

```bash
source config/paths.local.sh
python stage1-llava-ov/mdp3-motionbenc/eval_mdp3.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/mdp3_run" --num_frames 32 --pool-frames 32 --select-frames 8
python scripts/check_run.py "$HUVLLM_RESULTS/mdp3_run" \
    --expect-method mdp3 --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

The exact Carya submission is `stage1-llava-ov/mdp3-motionbenc/w2_run_mdp3.sbatch`.

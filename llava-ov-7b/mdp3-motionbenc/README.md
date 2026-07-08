# MDP3 × MotionBench Evaluation

Evaluates MDP3 on MotionBench using LLaVA-OV-7B (Qwen 1.5).

## What is MDP3?
MDP3 is a training-free method for efficient video LLM inference.

## Status
- **Smoke test**: PASSED (14/27 = 51.85%)
- **Full run**: Job 7662191 — RUNNING on compute-10-4
- **Current accuracy**: 53.25% (partial, ~50% data processed)

## Key Paths
| Path | Purpose |
|------|---------|
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | Model weights |
| `/project/rhu/dpalfaro/mdp3_pkgs/` | torch 2.3.1+cu121 (--target) |

## Files
- `run_mdp3.sbatch` — full eval (8,052 samples)
- `test_mdp3.sbatch` — smoke test (50 samples)
- `eval_mdp3.py` — custom evaluation script
- `setup_mdp3_env.sh` — env setup script

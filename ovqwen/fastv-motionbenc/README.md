# FastV × MotionBench Evaluation

Evaluates FastV on MotionBench using LLaVA-OV-7B (Qwen 1.5).

## What is FastV?
FastV prunes visual tokens based on attention scores. Paper evaluates on LLaVA-1.5-7B;
this evaluation uses LLaVA-OV-7B as an architectural extension.

## Key Paths
| Path | Purpose |
|------|---------|
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | Model weights |
| `/project/rhu/dpalfaro/code/FastV` | FastV source |

## Files
- `run_fastv.sbatch` — full eval
- `test_fastv.sbatch` — smoke test
- `eval_fastv.py` — custom evaluation script
- `strict/` — strict evaluation variants

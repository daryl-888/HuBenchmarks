# VisionZip × MotionBench Evaluation

Evaluates VisionZip on MotionBench using LLaVA-v1.5-7B.

## Results
| Config | Frames | Accuracy |
|--------|:------:|:--------:|
| dominant=54, contextual=10 | 8 | 40.09% |
| dominant=54, contextual=10 | 32 | 39.97% |

LLaVA-1.5-7B is a weak backbone — VisionZip paper also evaluates on LLaVA-NeXT and LLaVA-OV.

## Key Paths
| Path | Purpose |
|------|---------|
| LLaVA-1.5-7B weights | VisionZip loads its own backbone |

## Files
- `run_visionzip.sbatch` — full eval (8 frames)
- `run_visionzip_32f.sbatch` — full eval (32 frames)
- `test_visionzip.sbatch` — smoke test
- `eval_visionzip.py` — custom evaluation script

# fastvid on LLaVA-OV-7B (Qwen 1.5)

**⚠️ PORTED FROM: llava-ov-7b-qwen2/ → Needs verification run.**

This directory was created by copying sbatch/eval files from the Qwen2 version
and updating `--model-path` to `/project/rhu/dpalfaro/weights/llava-ov-7b`
and `--conv-template` to `qwen_1_5`.

Original fastvid evaluation was done on LLaVA-OV-7B-Qwen2 (`llava-ov-7b-qwen2`).
This port evaluates fastvid on LLaVA-OV-7B (Qwen 1.5) for cross-backbone comparison.

**Before running:** verify sbatch files have correct weights and conv template.

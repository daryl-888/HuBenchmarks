# dycoke on LLaVA-OV-7B-Qwen2 (Qwen2)

**⚠️ PORTED FROM: llava-ov-7b/ → Needs verification run.**

This directory was created by copying sbatch/eval files from the Qwen 1.5 version
and updating `--model-path` to `/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2`
and `--conv-template` to `qwen_2`.

Original dycoke evaluation was done on LLaVA-OV-7B (Qwen 1.5).
This port evaluates dycoke on LLaVA-OV-7B-Qwen2 (Qwen2) for cross-backbone comparison.

**Before running:** verify sbatch files have correct weights and conv template.

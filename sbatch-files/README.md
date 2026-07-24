# ⚠️ LEGACY — NOT DEPLOYED, DO NOT SUBMIT

These are pre-restructure sbatch templates kept for reference only.

`scripts/deploy.sh` syncs ONLY these trees to Carya:
`stage1-llava-ov/`, `stage2-llava-video/`, `stage3-qwen3-vl/`, `other-backbones/`.

This directory is **not** one of them, so nothing here is ever pushed. Several
files still reference the old flat paths (e.g.
`/project/rhu/dpalfaro/code/fastv-motionbenc/`) which no longer hold current code —
submitting one would run stale code or fail.

**Use the per-method sbatch inside the stage directories instead.**

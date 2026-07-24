#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HuVLLM path configuration — EDIT THIS FILE to replicate on your own system.
# ---------------------------------------------------------------------------
#
# Every sbatch / eval script in this repo was written for the UH Carya cluster,
# where paths are hardcoded as /project/rhu/dpalfaro/... . To reproduce our runs
# elsewhere, set the variables below to your own locations and `source` this file
# (or export the equivalents) before launching a job.
#
#     source config/paths.sh
#
# The defaults reproduce our exact environment; override any of them.
# ---------------------------------------------------------------------------

# Root of everything (weights, results, cache, conda envs, method source repos).
export HUVLLM_ROOT="${HUVLLM_ROOT:-/project/rhu/dpalfaro}"

# This repository's checkout on the compute node.
export HUVLLM_CODE="${HUVLLM_CODE:-$HUVLLM_ROOT/code}"

# Model weights. Each subdir is a HuggingFace snapshot (see docs/SETUP.md).
export HUVLLM_WEIGHTS="${HUVLLM_WEIGHTS:-$HUVLLM_ROOT/weights}"
export W_LLAVA_OV="${W_LLAVA_OV:-$HUVLLM_WEIGHTS/llava-ov-7b}"            # lmms-lab/llava-onevision-qwen2-7b-ov
export W_QWEN3VL="${W_QWEN3VL:-$HUVLLM_WEIGHTS/qwen3-vl-8b}"             # Qwen/Qwen3-VL-8B-Instruct
export W_LLAVA_VIDEO="${W_LLAVA_VIDEO:-$HUVLLM_WEIGHTS/llava-video-7b}"  # lmms-lab/LLaVA-Video-7B-Qwen2
export W_PLLAVA="${W_PLLAVA:-$HUVLLM_WEIGHTS/pllava-7b}"                 # ermu2001/pllava-7b (PEFT/LoRA)
export W_VICUNA="${W_VICUNA:-$HUVLLM_WEIGHTS/llava-v1.6-vicuna-7b}"      # liuhaotian/llava-v1.6-vicuna-7b
export W_LLAVA15="${W_LLAVA15:-$HUVLLM_WEIGHTS/llava-v1.5-7b}"           # liuhaotian/llava-v1.5-7b
export W_VIDEOITG="${W_VIDEOITG:-$HUVLLM_WEIGHTS/videoitg-8b}"           # nvidia/VideoITG-8B

# MotionBench dataset: video_info.meta.jsonl + self-collected/ + public-dataset/.
export MOTIONBENCH="${MOTIONBENCH:-/project/rhu/MotionBench_Data/MotionBench}"
export MOTIONBENCH_META="${MOTIONBENCH_META:-$MOTIONBENCH/video_info.meta.jsonl}"

# Where run outputs (results.jsonl + summary.json) are written.
export HUVLLM_RESULTS="${HUVLLM_RESULTS:-$HUVLLM_ROOT/results}"

# Conda envs (see docs/SETUP.md for the env matrix — methods need different ones).
export HUVLLM_ENVS="${HUVLLM_ENVS:-$HUVLLM_ROOT/conda/envs}"

# HuggingFace cache.
export HF_HOME="${HF_HOME:-$HUVLLM_ROOT/cache/huggingface}"

# Method source repositories (the authors' original code, cloned + patched).
# See patches/ for the diffs we applied.
export SRC_DYCOKE="${SRC_DYCOKE:-$HUVLLM_CODE/DyCoke}"
export SRC_HOLITOM="${SRC_HOLITOM:-$HUVLLM_CODE/HoliTom}"
export SRC_MDP3="${SRC_MDP3:-$HUVLLM_CODE/MDP3}"
export SRC_FASTV="${SRC_FASTV:-$HUVLLM_CODE/FastV}"
export SRC_VISIONZIP="${SRC_VISIONZIP:-$HUVLLM_CODE/VisionZip}"
export SRC_PRUNEVID="${SRC_PRUNEVID:-$HUVLLM_CODE/PruneVid}"
export SRC_DYTO="${SRC_DYTO:-$HUVLLM_ROOT/DYTO}"
export DYTO_ROOT="$SRC_DYTO"   # eval_dyto.py reads this for the llava alias

if [ "${HUVLLM_VERBOSE:-0}" = "1" ]; then
  echo "HuVLLM paths:"
  echo "  ROOT      = $HUVLLM_ROOT"
  echo "  WEIGHTS   = $HUVLLM_WEIGHTS"
  echo "  MOTIONBENCH = $MOTIONBENCH"
  echo "  RESULTS   = $HUVLLM_RESULTS"
fi

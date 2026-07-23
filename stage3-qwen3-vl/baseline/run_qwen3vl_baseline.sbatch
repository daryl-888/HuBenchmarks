#!/bin/bash
#SBATCH -J qwen3vl_baseline
#SBATCH -o /project/rhu/dpalfaro/results/qwen3vl_baseline_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/qwen3vl_baseline_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# Qwen3-VL-8B BASELINE — full MotionBench (8,052 samples), 32 frames, no method.
#
# This is Step 0 of the Stage-3 plan: a legitimate result in its own right AND the
# mandatory --vs-baseline reference for every Qwen3-VL method port. Without it no
# Stage-3 port can be proven to actually engage.
#
# Verified by smoke 7770277: 50% on 8 samples, real per-category output.
# Env `qwen3vl` is the ONLY env that can load Qwen3VLForConditionalGeneration
# (all others have transformers 4.45, which lacks the class).

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

OUT=/project/rhu/dpalfaro/results/qwen3vl_baseline_run1

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/baseline/eval_qwen3vl_baseline.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" || \
    echo "GATE FAILED - do not record this run"

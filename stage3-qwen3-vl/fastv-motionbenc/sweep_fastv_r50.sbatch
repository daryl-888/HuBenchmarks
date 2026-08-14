#!/bin/bash
#SBATCH -J s3_fastv_r50
#SBATCH -o /project/rhu/dpalfaro/results/s3_fastv_r50_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_fastv_r50_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# Retention sweep: fastv @ retention 0.50 on Qwen3-VL-8B -- completes FastV's
# full published range (10/15/20/25/50/75%) on this backbone, matching the
# LLaVA-OV sweep (Table 3).
#
# NOTE for FastV: --fastv_r is the DROP fraction, so retention 0.50 -> --fastv_r 0.50.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/s3_fastv_r50_run

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/fastv-motionbenc/eval_fastv_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --fastv --fastv_k 2 --fastv_r 0.50

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method fastv \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 || echo "GATE FAILED - do not record"

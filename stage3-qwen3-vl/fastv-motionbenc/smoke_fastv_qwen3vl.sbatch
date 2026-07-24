#!/bin/bash
#SBATCH -J s3_fastv_smoke
#SBATCH -o /project/rhu/dpalfaro/results/s3_fastv_smoke_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_fastv_smoke_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 0:40:00

# Wave 3 #1 — FastV ported to Qwen3-VL (first REAL Stage-3 method).
# Must print "FastV(Qwen3-VL) ACTIVE: visual=N keep=M" or it did not engage.
# Env qwen3vl is the only one that can load Qwen3VLForConditionalGeneration.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

OUT=/project/rhu/dpalfaro/results/s3_fastv_smoke_${SLURM_JOB_ID}

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/fastv-motionbenc/eval_fastv_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --fastv --fastv_k 2 --fastv_r 0.85 \
    --limit 8

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method fastv --smoke || echo "GATE FAILED - do not record"

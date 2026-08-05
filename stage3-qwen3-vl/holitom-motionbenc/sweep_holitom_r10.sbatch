#!/bin/bash
#SBATCH -J s3_holitom_r10
#SBATCH -o /project/rhu/dpalfaro/results/s3_holitom_r10_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_holitom_r10_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# Retention sweep: holitom @ retention 0.1 on Qwen3-VL-8B.
# Only the retention knob varies; 32 frames and every other parameter are held
# fixed, so differences across r=0.10/0.15/0.25 are attributable to retention.
#
# NOTE for FastV: --fastv_r is the DROP fraction, so retention r maps to
# --fastv_r (1-r). retention 0.10 -> --fastv_r 0.9.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/s3_holitom_r10_run

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/holitom-motionbenc/eval_holitom_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --holitom --retain_ratio 0.1 --temporal_t 0.80 --holitom_k 18 --holitom_r 0.5

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method holitom \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 || echo "GATE FAILED - do not record"

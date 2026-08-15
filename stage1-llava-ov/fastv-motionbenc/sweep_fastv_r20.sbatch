#!/bin/bash
#SBATCH -J s1_fastv_r20
#SBATCH -o /project/rhu/dpalfaro/results/s1_fastv_r20_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_fastv_r20_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# Retention sweep: fastv @ retention 0.20 on LLaVA-OV-7B -- completes the
# 20% column in Table 3 (tab:sweepllava), which was missing on this backbone
# even though FlashVID/HoliTom both have it here.
#
# NOTE for FastV: --fastv_r is the DROP fraction, so retention r maps to
# --fastv_r (1-r). retention 0.20 -> --fastv_r 0.80.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/s1_fastv_r20_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/fastv-motionbenc/eval_fastv.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --fastv --fastv_k 2 --fastv_r 0.80

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method fastv \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

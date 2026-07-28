#!/bin/bash
#SBATCH -J s1_flashvid_r25
#SBATCH -o /project/rhu/dpalfaro/results/s1_flashvid_r25_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_flashvid_r25_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# Retention sweep: flashvid @ retention 0.25 on LLaVA-OV-7B.
# Only the retention knob varies; 32 frames and every other parameter are held
# fixed, so differences across r=0.10/0.15/0.25 are attributable to retention.
#
# NOTE for FastV: --fastv_r is the DROP fraction, so retention r maps to
# --fastv_r (1-r). retention 0.10 -> --fastv_r 0.9.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/s1_flashvid_r25_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/flashvid-motionbenc/eval_flashvid.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --flashvid --retention_ratio 0.25 --alpha 0.7 --temporal_threshold 0.8

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method flashvid \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

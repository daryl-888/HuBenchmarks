#!/bin/bash
#SBATCH -J s1_fastv_r75
#SBATCH -o /project/rhu/dpalfaro/results/s1_fastv_r75_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_fastv_r75_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# Threshold probe: FastV @ keep 75% on LLaVA-OV-7B.
# Only the retention knob varies; 32 frames and every other parameter are held
# fixed, so differences across keep 10/15/25/50/75/100% are attributable to it.
#
# NOTE for FastV: --fastv_r is the DROP fraction, so keeping 75% of
# visual tokens means --fastv_r 0.25.
#
# Why this run exists: FastV collapses to ~36-37% at keep 10/15/25% on
# LLaVA-OV and recovers to the backbone at keep 100%, so the failure
# threshold lies in the untested 25-100% interval. The authors' released
# sweep (rank_list=(72 144 288 432) of 576 tokens) covers keep 12.5-75%,
# so 75% is one of FastV's own published settings. See
# docs/RETENTION_DIAGNOSIS.md.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/s1_fastv_r75_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/fastv-motionbenc/eval_fastv.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --fastv --fastv_k 2 --fastv_r 0.25

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method fastv \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

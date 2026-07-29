#!/bin/bash
#SBATCH -J samp_mdp3
#SBATCH -o /project/rhu/dpalfaro/results/samp_mdp3_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/samp_mdp3_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# SAMPLED run: mdp3 @ temperature 0.7, top_p 0.9, seed 0, 32 frames.
#
# NOT a gated result. Sampling breaks the divergence check (two sampled runs
# differ by chance), so this cannot prove a method engaged and must never be
# merged into master-results.md. Purpose: A/B/C/D answer distribution.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/samp_mdp3_run

/project/rhu/dpalfaro/conda/envs/mdp3/bin/python3 \
    /project/rhu/dpalfaro/code/sampled-llava-ov/eval_mdp3_sampled.py \
    --meta-file /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --video-roots /project/rhu/MotionBench_Data/MotionBench \
    --model-path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --output-dir ${OUT} \
    --pool-frames 32 --select-frames 8 \
    --temperature 0.7 --top_p 0.9 --seed 0

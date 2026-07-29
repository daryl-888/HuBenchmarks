#!/bin/bash
# VideoITG Stage 2: LLaVA-OV-7B inference with VideoITG-selected frames.
# Requires grounding.sbatch to have completed first.
#SBATCH -J samp_videoitg
#SBATCH -o /project/rhu/dpalfaro/results/samp_videoitg_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/samp_videoitg_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DyCoke repo contains the llava package (LLaVA-NeXT source)
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:$PYTHONPATH

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/sampled-llava-ov/eval_videoitg_sampled.py \
    --model_path      /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path       /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --grounding_jsonl /project/rhu/dpalfaro/results/videoitg_grounding/frame_scores.jsonl \
    --output_dir /project/rhu/dpalfaro/results/samp_videoitg_run \
    --num_frames      32

OUT=/project/rhu/dpalfaro/results/samp_videoitg_run
echo "=== GATE: did videoitg actually engage? ==="
# NO GATE: sampling breaks the divergence check (two sampled runs differ by
# chance), so --vs-baseline cannot prove engagement here. Distribution analysis
# is done afterwards by sampled-llava-ov/build_distribution.py.
    --temperature 0.7 --top_p 0.9 --seed 0

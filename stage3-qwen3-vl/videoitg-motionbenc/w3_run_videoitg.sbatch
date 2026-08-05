#!/bin/bash
#SBATCH -J w3_videoitg
#SBATCH -o /project/rhu/dpalfaro/results/w3_videoitg_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/w3_videoitg_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00

# Wave 3 #1 — videoitg ported to Qwen3-VL (Wave 3).
# Must print "videoitg(Qwen3-VL) ACTIVE: ..." or it did not engage.
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

OUT=/project/rhu/dpalfaro/results/w3_videoitg_run

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/videoitg-motionbenc/eval_videoitg_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/w3_videoitg_run \
    --num_frames 32 \
    --grounding_jsonl /project/rhu/dpalfaro/results/videoitg_grounding/frame_scores.jsonl

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method videoitg \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 || echo "GATE FAILED - do not record"

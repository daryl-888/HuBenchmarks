#!/bin/bash
#SBATCH -J w3_sttm
#SBATCH -o /project/rhu/dpalfaro/results/w3_sttm_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/w3_sttm_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 20:00:00

# STTM ported to Qwen3-VL.
#
# This port is the riskiest of the set: unlike the other eight it SHORTENS the
# sequence instead of masking, so position_ids, visual_pos_masks and
# deepstack_visual_embeds are all rebuilt. Two distinct failure modes to watch:
#   * "NO REDUCTION" in the log -> quadtree merged nothing (threshold too high)
#   * ACTIVE printed but 0 divergence -> the rebuild was ignored somewhere
# Only ACTIVE *and* nonzero divergence counts as engaged.
#
# PYTHONPATH must reach the STTM checkout for token_merging_utils.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SRC_STTM=/project/rhu/dpalfaro/code/STTM
export PYTHONPATH=${SRC_STTM}:${PYTHONPATH}

cd /project/rhu/dpalfaro

OUT=/project/rhu/dpalfaro/results/w3_sttm_run

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/sttm-motionbenc/eval_sttm_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --sttm --tree_thresh 0.85 --temporal_thresh -1.0 --start_layer 2

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method sttm \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 || echo "GATE FAILED - do not record"

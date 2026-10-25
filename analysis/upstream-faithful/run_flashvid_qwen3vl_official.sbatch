#!/bin/bash
#SBATCH -J s3_flashvid_off
#SBATCH -o /project/rhu/dpalfaro/results/s3_flashvid_official_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_flashvid_official_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00

# FlashVID on Qwen3-VL using the AUTHORS' implementation.
#
# The gated w3_flashvid_run hand-reimplements the scoring rule and never
# imports the flashvid package, so it omits inner-LLM compression
# (pruning_layer, llm_retention_ratio), expansion, and the segment
# parameters. FlashVID ships native Qwen3-VL support, so this run calls
# flashvid() with the settings from FlashVID/scripts/qwen3_vl.sh.
# Must print "FlashVID OFFICIAL: ..." or it did not engage.
# See docs/UPSTREAM_CROSSREF.md section 2.
# Env qwen3vl is the only one that can load Qwen3VLForConditionalGeneration.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SRC_FLASHVID=/project/rhu/dpalfaro/code/FlashVID

cd /project/rhu/dpalfaro

OUT=/project/rhu/dpalfaro/results/s3_flashvid_official_run

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/analysis/upstream-faithful/eval_flashvid_qwen3vl_official.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/s3_flashvid_official_run \
    --num_frames 32 \
    --flashvid --retention_ratio 0.15 --alpha 0.7 --temporal_threshold 0.8 \
    --token_selection_method attn_div --min_segment_num 4 --segment_threshold 0.9 \
    --expansion 1.25 --pruning_layer 28 --llm_retention_ratio 0.1

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method flashvid \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 || echo "GATE FAILED - do not record"

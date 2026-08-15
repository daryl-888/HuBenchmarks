#!/bin/bash
#SBATCH -J s3_flashvid_off_r25
#SBATCH -o /project/rhu/dpalfaro/results/s3_flashvid_official_r25_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_flashvid_official_r25_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00

# FlashVID on Qwen3-VL @ retention 0.25, AUTHORS' implementation (matches the
# verified 15%/20% cells, s3_flashvid_official_run / s3_flashvid_official_r20_run)
# -- NOT the simplified stage3-qwen3-vl/eval_flashvid_qwen3vl.py used for the
# old r10/r25 sweep points, which omits inner-LLM compression and is not
# comparable to this cell. Only retention_ratio changes vs.
# run_flashvid_qwen3vl_official_v2.sbatch.
# Env flashvid_qwen3vl_official (transformers==4.57.3) is required -- the
# shared qwen3vl env crashes FlashVID's inner-LLM pruning stage.

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

OUT=/project/rhu/dpalfaro/results/s3_flashvid_official_r25_run

/project/rhu/dpalfaro/conda/envs/flashvid_qwen3vl_official/bin/python3 \
    /project/rhu/dpalfaro/code/analysis/upstream-faithful/eval_flashvid_qwen3vl_official.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --flashvid --retention_ratio 0.25 --alpha 0.7 --temporal_threshold 0.8 \
    --token_selection_method attn_div --min_segment_num 4 --segment_threshold 0.9 \
    --expansion 1.25 --pruning_layer 28 --llm_retention_ratio 0.1

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method flashvid \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 || echo "GATE FAILED - do not record"

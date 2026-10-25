#!/bin/bash
#SBATCH -J s3_sttm_pub
#SBATCH -o /project/rhu/dpalfaro/results/s3_sttm_pub_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_sttm_pub_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu
#
# STTM on Qwen3-VL at a PUBLISHED configuration.
#
# The gated w3_sttm_run used --root_level 0 --temporal_thresh -1.0. Neither
# appears in the authors' scripts/eval/run_vidqa.sh, which defines:
#   sttm_llava_common_cfg   = --sa_start_layer_idx 2 --sa_tree_root_level 1
#   sttm_qwen2vl_common_cfg = --sa_start_layer_idx 2 --sa_tree_root_level 2
# and pairs those with --sa_tree_thresh / --sa_tree_temporal_thresh per
# benchmark and budget (e.g. 0.85 / 0.65 at the 50% budget).
#
# root_level=0 is off-spec and temporal_thresh=-1.0 disables temporal merging
# outright, making the gated cell a spatial-only variant the authors never run.
# Qwen3-VL is Qwen-family, so this uses root_level=2 with thresh 0.85 /
# temporal 0.65. See docs/UPSTREAM_CROSSREF.md section 3.
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

OUT=/project/rhu/dpalfaro/results/s3_sttm_pub_run

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/sttm-motionbenc/eval_sttm_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --sttm --tree_thresh 0.85 --temporal_thresh 0.65 --root_level 2 --start_layer 2

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method sttm \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 || echo "GATE FAILED - do not record"

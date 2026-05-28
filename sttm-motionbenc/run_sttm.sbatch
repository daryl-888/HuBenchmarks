#!/bin/bash
#SBATCH -J sttm_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/sttm_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/sttm_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/project/rhu/dpalfaro/code/STTM:$PYTHONPATH

cd /project/rhu/dpalfaro/code/STTM

echo "=== Step 1: Extract visual features ==="
/project/rhu/dpalfaro/conda/envs/sttm/bin/python3 \
    /project/rhu/dpalfaro/code/sttm-motionbenc/extract_features.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --video_base /project/rhu/MotionBench_Data/MotionBench \
    --output_dir /project/rhu/dpalfaro/sttm_features/motionbench \
    --num_frames 32

echo "=== Step 2: STTM inference ==="
/project/rhu/dpalfaro/conda/envs/sttm/bin/python3 \
    /project/rhu/dpalfaro/code/sttm-motionbenc/eval_motionbench.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --features_dir /project/rhu/dpalfaro/sttm_features/motionbench \
    --output_dir /project/rhu/dpalfaro/results/sttm_run1 \
    --sa_start_layer_idx 2 \
    --sa_tree_thresh 0.85 \
    --sa_tree_temporal_thresh 0.65 \
    --sa_tree_root_level 1

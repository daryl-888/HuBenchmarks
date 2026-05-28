#!/bin/bash
#SBATCH -J prunevid_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/prunevid_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/prunevid_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

# Run from PruneVid repo root so its tasks/ package is on the path
export PYTHONPATH=/project/rhu/dpalfaro/code/PruneVid:$PYTHONPATH
cd /project/rhu/dpalfaro/code/PruneVid

# PruneVid params (from PruneVid paper / scripts/eval.sh defaults):
#   selected_layer 10     LLM layer whose attention scores drive query-aware pruning
#   alpha 0.4             pruning intensity (lower = more tokens retained)
#   tau 0.8               attention threshold for relevant token selection
#   temporal_segment_ratio 0.25   fraction of temporal divisions for stage-1 merging
#   cluster_ratio 0.5     spatial clustering ratio for stage-1 merging
/project/rhu/dpalfaro/conda/envs/prunevid/bin/python3 \
    /project/rhu/dpalfaro/code/prunevid-motionbenc/eval_motionbench.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --video_base /project/rhu/MotionBench_Data/MotionBench \
    --output_dir /project/rhu/dpalfaro/results/prunevid_run1 \
    --num_frames 32 \
    --selected_layer 10 \
    --alpha 0.4 \
    --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5

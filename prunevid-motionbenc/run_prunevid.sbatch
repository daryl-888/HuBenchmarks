#!/bin/bash
#SBATCH -J prunevid_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/prunevid_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/prunevid_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/PruneVid:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/prunevid/bin/python3 \
    /project/rhu/dpalfaro/code/prunevid-motionbenc/eval_prunevid.py \
    --model_path /project/rhu/dpalfaro/weights/pllava-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/prunevid_run1 \
    --num_frames 16 \
    --cluster_ratio 0.5 \
    --temporal_segment_ratio 0.25 \
    --selected_layer 10 \
    --alpha 0.4 \
    --tau 0.8

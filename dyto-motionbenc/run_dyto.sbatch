#!/bin/bash
#SBATCH -J dyto_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/dyto_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dyto_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

# DYTO/dyto/llava must come first — it ships a patched LLaVA-NeXT
export PYTHONPATH=/project/rhu/dpalfaro/code/DYTO/dyto/llava:/project/rhu/dpalfaro/code/DYTO:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

# DyTo token merging enabled
TEMPORAL_AGGREGATION=cluster \
/project/rhu/dpalfaro/conda/envs/dyto/bin/python3 \
    /project/rhu/dpalfaro/code/dyto-motionbenc/eval_dyto.py \
    --model-path /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --meta-file  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output-dir /project/rhu/dpalfaro/results/dyto_run1 \
    --num-frames 32
#!/bin/bash
#SBATCH -J dyto_test
#SBATCH -o /project/rhu/dpalfaro/results/dyto_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dyto_test_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

export PYTHONPATH=/project/rhu/dpalfaro/code/DYTO/dyto/llava:/project/rhu/dpalfaro/code/DYTO:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

# Smoke test: 50 samples, DyTo token merging enabled
TEMPORAL_AGGREGATION=cluster \
/project/rhu/dpalfaro/conda/envs/dyto/bin/python3 \
    /project/rhu/dpalfaro/code/dyto-motionbenc/eval_dyto.py \
    --model-path /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --meta-file  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output-dir /project/rhu/dpalfaro/results/dyto_test_${SLURM_JOB_ID} \
    --num-frames 32 \
    --limit 50
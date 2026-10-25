#!/bin/bash
#SBATCH -J ob_trajvit_run
#SBATCH -o /project/rhu/dpalfaro/results/ob_trajvit_run_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ob_trajvit_run_%j.err
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
export PYTHONPATH=/project/rhu/dpalfaro/code/TrajViT:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

# TODO: TrajViT uses a trained trajectory-aware encoder — not plug-and-play on LLaVA-OV-7B.
# Confirm whether TrajViT provides its own lmms_eval backend or requires a custom model wrapper.
# May need a separate conda env and its own backbone weights.
/project/rhu/dpalfaro/conda/envs/trajvit/bin/python3 -m lmms_eval \
    --include_path /project/rhu/dpalfaro/code/trajvit-motionbenc/tasks \
    --model trajvit \
    --model_args pretrained=/project/rhu/dpalfaro/weights/trajvit \
    --tasks motionbench \
    --batch_size 1 \
    --output_path /project/rhu/dpalfaro/results/trajvit_run1 \
    --log_samples

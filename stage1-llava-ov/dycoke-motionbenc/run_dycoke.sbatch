#!/bin/bash
#SBATCH -J s1_dycoke_run
#SBATCH -o /project/rhu/dpalfaro/results/s1_dycoke_run_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_dycoke_run_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/dycoke-motionbenc/eval_dycoke.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --conv_template qwen_2 \
    --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/stage1_dycoke_run1 \
    --num_frames 32

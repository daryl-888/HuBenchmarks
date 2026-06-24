#!/bin/bash
#SBATCH -J flashvid_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/flashvid_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/flashvid_%j.err
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
export WORLD_SIZE=1
export PYTHONPATH=/project/rhu/dpalfaro/code/FlashVID:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/flashvid/bin/python3 \
    /project/rhu/dpalfaro/code/flashvid-motionbenc/eval_flashvid_motionbench.py \
    --meta-file    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --video-roots  /project/rhu/MotionBench_Data/MotionBench \
    --model-path   /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 \
    --output-dir   /project/rhu/dpalfaro/results/flashvid_run4 \
    --frame-counts 8 \
    --conv-template qwen_2 \
    --enable-flashvid \
    --retention-ratio 0.25 \
    --alpha 0.7 \
    --temporal-threshold 0.8 \
    --prompt-style strict \
    --max-new-tokens 16

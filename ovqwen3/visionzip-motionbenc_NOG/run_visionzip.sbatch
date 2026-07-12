#!/bin/bash
#SBATCH -J ovqwen3_visionzip_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/visionzip_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/visionzip_%j.err
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
# VisionZip patches CLIPVisionTower — requires liuhaotian/LLaVA (not LLaVA-NeXT).
# Clone: git clone https://github.com/haotian-liu/LLaVA && pip install -e LLaVA/
export PYTHONPATH=${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Weights: download LLaVA-1.5-7B if not present
# huggingface-cli download liuhaotian/llava-v1.5-7b \
#     --local-dir /project/rhu/dpalfaro/weights/qwen3-vl-8b

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/visionzip/bin/python3 \
    /project/rhu/dpalfaro/code/visionzip-motionbenc/eval_visionzip.py \
    --model_path   /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --model_name   llava_v1.5_7b \
     \
    --meta_path    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir   /project/rhu/dpalfaro/results/visionzip_run1 \
    --num_frames 32 \
    --dominant     54 \
    --contextual   10

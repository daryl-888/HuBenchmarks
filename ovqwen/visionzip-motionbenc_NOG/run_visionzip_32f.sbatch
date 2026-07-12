#!/bin/bash
#SBATCH -J ovqwen15_visionzip_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/visionzip_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/visionzip_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# 32-frame run. dominant=54, contextual=10 → 64 tokens/frame × 32 frames = 2048 visual tokens.
# Without compression, 32 frames × 256 CLIP patches = 8192 tokens (blows LLaVA-1.5's context).
# VisionZip compression makes 32 frames feasible here.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/VisionZip:/project/rhu/dpalfaro/code/LLaVA:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/visionzip/bin/python3 \
    /project/rhu/dpalfaro/code/visionzip-motionbenc/eval_visionzip.py \
    --model_path   /project/rhu/dpalfaro/weights/llava-ov-7b \
    --model_name   qwen_1_5.5_7b \
    --conv_template qwen_1_5 \
    --meta_path    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir   /project/rhu/dpalfaro/results/visionzip_run2_32f \
    --num_frames   32 \
    --dominant     54 \
    --contextual   10

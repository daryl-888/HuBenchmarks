#!/bin/bash
#SBATCH -J visionzip_ovqwen2
#SBATCH -o /project/rhu/dpalfaro/results/visionzip_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/visionzip_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
# visionzip module is at /project/rhu/dpalfaro/code/VisionZip/visionzip/
# llava module is at /project/rhu/dpalfaro/AIM/llava/ (for load_pretrained_model)
export PYTHONPATH=/project/rhu/dpalfaro/code/VisionZip:/project/rhu/dpalfaro/AIM:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/visionzip/bin/python3 \
    /project/rhu/dpalfaro/code/visionzip-motionbenc/eval_visionzip.py \
    --model_path   /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 \
    --model_name   llava_qwen \
    --conv_template qwen_2 \
    --meta_path    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir   /project/rhu/dpalfaro/results/visionzip_ovqwen2 \
    --num_frames   32 \
    --dominant     54 \
    --contextual   10

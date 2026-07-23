#!/bin/bash
#SBATCH -J holitom_ovqwen15
#SBATCH -o /project/rhu/dpalfaro/results/holitom_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/holitom_%j.err
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
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use holitom conda env (has transformers 4.45.2 with rope_utils)
# PYTHONPATH: DyCoke for llava builder, HoliTom for monkey-patch code
# Exclude AIM's transformers path to avoid version mismatch
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:/project/rhu/dpalfaro/code/HoliTom
export WRAPPER=holitom
export RETAIN_RATIO=0.15
export T=0.80
export HOLITOM_k=18
export HOLITOM_r=0.5

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/holitom/bin/python3 \
    /project/rhu/dpalfaro/code/ovqwen/holitom-motionbenc/eval_holitom.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/holitom_ovqwen15_fresh \
    --num_frames 32

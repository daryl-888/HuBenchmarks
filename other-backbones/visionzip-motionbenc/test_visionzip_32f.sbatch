#!/bin/bash
#SBATCH -J visionzip_32f_test
#SBATCH -o /project/rhu/dpalfaro/results/visionzip_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/visionzip_test_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# Smoke test — 50 samples, 32 frames.

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
    /project/rhu/dpalfaro/code/other-backbones/visionzip-motionbenc/eval_visionzip.py \
    --model_path   /project/rhu/dpalfaro/weights/llava-v1.5-7b \
    --model_name   llava_v1.5_7b \
    --conv_template llava_v1 \
    --meta_path    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir   /project/rhu/dpalfaro/results/visionzip_test_32f_${SLURM_JOB_ID} \
    --num_frames   32 \
    --dominant     54 \
    --contextual   10 \
    --limit        50

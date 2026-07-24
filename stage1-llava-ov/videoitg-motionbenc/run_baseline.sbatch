#!/bin/bash
#SBATCH -J ovqwen15_videoitg_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/baseline_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/baseline_%j.err
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
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 -m lmms_eval \
    --model llava_onevision \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b --conv_template qwen_1_5 --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl --num_frames 32 \
    --output_dir /project/rhu/dpalfaro/results/ovqwen_videoitg_run1
    --output_path /project/rhu/dpalfaro/results/baseline_run1 \
    

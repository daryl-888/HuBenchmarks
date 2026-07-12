#!/bin/bash
# VideoITG Stage 1 test — 50 samples
#SBATCH -J ovqwen15_videoitg_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/videoitg_test_grounding_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/videoitg_test_grounding_%j.err
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
export PYTHONPATH=/project/rhu/dpalfaro/code/VideoITG:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/videoitg/bin/python3 \
    /project/rhu/dpalfaro/code/videoitg-motionbenc/eval_videoitg_grounding.py \
    --model_path        /project/rhu/dpalfaro/weights/videoitg-8b \
    --meta_path         /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir        /project/rhu/dpalfaro/results/videoitg_test_grounding_${SLURM_JOB_ID} \
    --num_frames_sample 512 \
    --num_frames_select 32 \
    --target_fps        2 \
    --limit 50

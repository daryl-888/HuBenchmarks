#!/bin/bash
# VideoITG Stage 1: Frame Grounding
# Scores 512 uniformly sampled frames per video using the VideoITG-8B selector model.
# Outputs a JSONL with per-sample frame scores → used by run_videoitg.sbatch (Stage 2).
#
# Run this first. Once complete, submit run_videoitg.sbatch.
#SBATCH -J videoitg_grounding
#SBATCH -o /project/rhu/dpalfaro/results/videoitg_grounding_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/videoitg_grounding_%j.err
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
export PYTHONPATH=/project/rhu/dpalfaro/code/VideoITG:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

# Stage 1: VideoITG-8B scores 512 frames per video for relevance to the question.
# num_frames=512: uniformly sampled frames to score
# target_fps=1: frame extraction rate
# output_dir: where per-sample frame-score JSONL lands (input to Stage 2)
/project/rhu/dpalfaro/conda/envs/videoitg/bin/python3 -m lmms_eval \
    --model videoitg \
    --model_args pretrained=/project/rhu/dpalfaro/weights/videoitg-8b,num_frames=512,target_fps=1,output_dir=/project/rhu/dpalfaro/results/videoitg_grounding \
    --include_path /project/rhu/dpalfaro/code/videoitg-motionbenc/tasks \
    --tasks motionbench \
    --batch_size 1 \
    --output_path /project/rhu/dpalfaro/results/videoitg_grounding_lmms

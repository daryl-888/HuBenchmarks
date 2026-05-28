#!/bin/bash
# VideoITG Stage 2: Downstream LLM Inference
# Reads the top-K frame indices selected by Stage 1 (grounding.sbatch) and
# feeds them to LLaVA-OV-7B for MotionBench QA.
#
# Requires grounding.sbatch to have completed first.
# Update GROUNDING_JSONL below with the actual path from Stage 1 output.
#SBATCH -J videoitg_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/videoitg_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/videoitg_%j.err
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

# Path to the frame-score JSONL produced by grounding.sbatch (Stage 1).
# Check /project/rhu/dpalfaro/results/videoitg_grounding/ for the file name.
GROUNDING_JSONL=/project/rhu/dpalfaro/results/videoitg_grounding/motionbench_frame_scores.jsonl

# Stage 2: LLaVA-OV-7B with VideoITG-selected frames.
# frame_indices_jsonl: top-K frame indices from Stage 1 (re-ordered chronologically)
# num_frame=32: number of frames passed to the downstream LLM
/project/rhu/dpalfaro/conda/envs/videoitg/bin/python3 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen,frame_indices_jsonl=${GROUNDING_JSONL},num_frame=32 \
    --include_path /project/rhu/dpalfaro/code/videoitg-motionbenc/tasks \
    --tasks motionbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videoitg \
    --output_path /project/rhu/dpalfaro/results/videoitg_run1

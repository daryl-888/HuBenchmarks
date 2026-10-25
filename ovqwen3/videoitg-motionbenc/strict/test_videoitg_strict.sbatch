#!/bin/bash
# VideoITG Stage 2 test — 50 samples. Run after test_grounding.sbatch.
# Update GROUNDING_JSONL with the job ID from the grounding step.
#SBATCH -J videoitg_test
#SBATCH -o /project/rhu/dpalfaro/results/videoitg_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/videoitg_test_%j.err
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
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:$PYTHONPATH

cd /project/rhu/dpalfaro

# Pass GROUNDING_JOBID via: sbatch --export=ALL,GROUNDING_JOBID=<jobid> test_videoitg.sbatch
GROUNDING_JSONL=/project/rhu/dpalfaro/results/videoitg_test_grounding_${GROUNDING_JOBID}/frame_scores.jsonl

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/videoitg-motionbenc/eval_videoitg_infer.py \
    --model_path      /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path       /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --grounding_jsonl ${GROUNDING_JSONL} \
    --output_dir      /project/rhu/dpalfaro/results/videoitg_test_${SLURM_JOB_ID} \
    --num_frames      32 \
    --limit           50

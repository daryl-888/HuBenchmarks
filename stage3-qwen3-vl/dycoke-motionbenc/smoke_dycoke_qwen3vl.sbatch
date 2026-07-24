#!/bin/bash
#SBATCH -J s3_dycoke_smoke
#SBATCH -o /project/rhu/dpalfaro/results/s3_dycoke_smoke_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_dycoke_smoke_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 0:40:00

# Wave 3 #1 — dycoke ported to Qwen3-VL (Wave 3).
# Must print "dycoke(Qwen3-VL) ACTIVE: ..." or it did not engage.
# Env qwen3vl is the only one that can load Qwen3VLForConditionalGeneration.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

OUT=/project/rhu/dpalfaro/results/s3_dycoke_smoke_${SLURM_JOB_ID}

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/dycoke-motionbenc/eval_dycoke_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --dycoke --dycoke_l 3 --dycoke_p 0.7 --dycoke_k 0.7 \
    --limit 8

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method dycoke \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 --smoke || echo "GATE FAILED - do not record"

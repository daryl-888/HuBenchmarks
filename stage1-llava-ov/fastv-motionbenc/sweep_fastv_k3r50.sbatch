#!/bin/bash
#SBATCH -J s1_fastv_k3r50
#SBATCH -o /project/rhu/dpalfaro/results/s1_fastv_k3r50_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_fastv_k3r50_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# FastV at the OFFICIAL repo's most-reported setting: K=3, R=50%.
#
# We cloned chenllliang/FastV (the first author's personal copy) at f95102a,
# 2024-03-20. The canonical repo is pkunlp-icler/FastV [ECCV 2024 Oral], which
# demonstrates K=2 with R=25-75% and reports K=3 / R=50% most often. Our gated
# run used K=2 at R=85% (keep 15%) -- outside that range on both axes.
#
# Neither repo ships a video evaluation, so no video setting is authoritative;
# this is the closest thing to "FastV as its authors report it".
# See docs/UPSTREAM_CROSSREF.md section 3.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/s1_fastv_k3r50_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/fastv-motionbenc/eval_fastv.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --fastv --fastv_k 3 --fastv_r 0.50

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method fastv \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

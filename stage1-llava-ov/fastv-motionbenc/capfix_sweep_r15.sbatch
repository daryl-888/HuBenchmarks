#!/bin/bash
#SBATCH -J capfix_fv_r15
#SBATCH -o /project/rhu/dpalfaro/results/capfix_fastv_r15_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/capfix_fastv_r15_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# FastV @ 15% retention on LLaVA-OV-7B, WITH the PrunableDynamicCache fix
# (physical-prune + no-legacy-cache-roundtrip). Replaces the pre-fix
# s1_fastv_r15_run / w2_fastv_run point, which never actually pruned
# anything (silently discarded via a legacy-cache reconstruction).

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/capfix_fastv_r15_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/fastv-motionbenc/eval_fastv_capfix.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --fastv --fastv_k 2 --fastv_r 0.85

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method fastv \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

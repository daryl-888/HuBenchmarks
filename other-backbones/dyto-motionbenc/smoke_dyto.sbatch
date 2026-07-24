#!/bin/bash
#SBATCH -J ob_dyto_smoke
#SBATCH -o /project/rhu/dpalfaro/results/ob_dyto_smoke_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ob_dyto_smoke_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 0:40:00
module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/dyto_pkgs:/project/rhu/dpalfaro/code/DYTO:${PYTHONPATH:-}
export DYTO_FINCH_SHIM=1   # LABELLED VARIANT: standard FINCH, not TW-FINCH
export DYTO_ROOT=/project/rhu/dpalfaro/code/DYTO
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/ob_dyto_smoke_${SLURM_JOB_ID}
/project/rhu/dpalfaro/conda/envs/dyto/bin/python3 \
    /project/rhu/dpalfaro/code/other-backbones/dyto-motionbenc/eval_dyto.py \
    --model-path /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --meta-path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output-dir ${OUT} \
    --num-frames 32 \
    --limit 8
echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" --expect-method dyto --smoke || echo "GATE FAILED"

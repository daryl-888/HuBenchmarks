#!/bin/bash
#SBATCH -J vz_ov_r20
#SBATCH -o /project/rhu/dpalfaro/results/visionzip_ov_r20_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/visionzip_ov_r20_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# VisionZip (contextual-only) on LLaVA-OV-7B @ 20% retention. New port
# (SigLIP has no CLS token, so only the contextual half is implementable --
# same limitation as the existing Qwen3-VL cell in Table 2).

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/visionzip_ov_r20_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/visionzip-motionbenc/eval_visionzip_ov.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --visionzip --retention_ratio 0.2

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method visionzip \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

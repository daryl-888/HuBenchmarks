#!/bin/bash
#SBATCH -J w2_flashvid
#SBATCH -o /project/rhu/dpalfaro/results/w2_flashvid_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/w2_flashvid_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export WORLD_SIZE=1
export PYTHONPATH=/project/rhu/dpalfaro/code/FlashVID:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/flashvid/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/flashvid-motionbenc/eval_flashvid.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/w2_flashvid_run \
    --num_frames 32 \
    --conv_template qwen_2 \
    --retention_ratio 0.15 \
    --alpha 0.7 \
    --temporal_threshold 0.8

OUT=/project/rhu/dpalfaro/results/w2_flashvid_run
echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "$OUT" \
    --expect-method flashvid || echo "GATE FAILED - do not record"

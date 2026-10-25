#!/bin/bash
#SBATCH -J smoke_fastv_capfix
#SBATCH -o /project/rhu/dpalfaro/results/smoke_fastv_capfix_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/smoke_fastv_capfix_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 2:00:00

# Smoke test for the PrunableDynamicCache kv_cache fix (see paper.tex Notable /
# results.tex discussion). Runs FastV at its standard 15% keep (K=2, R=0.85) on
# a 300-sample slice only, to confirm the fix moves accuracy out of the
# ~36-37% collapse band before committing the full 6-point sweep re-run.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/smoke_fastv_capfix_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/fastv-motionbenc/eval_fastv_capfix.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 --limit 300 \
    --fastv --fastv_k 2 --fastv_r 0.85

echo "=== SMOKE RESULT ==="
python3 -c "
import json
s = json.load(open('${OUT}/summary.json'))
print('accuracy:', s['accuracy'])
print('correct:', s['correct'], '/', s['total_scoreable'])
"

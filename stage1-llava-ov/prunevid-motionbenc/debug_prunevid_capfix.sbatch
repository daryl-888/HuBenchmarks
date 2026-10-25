#!/bin/bash
#SBATCH -J dbg_prunevid_capfix
#SBATCH -o /project/rhu/dpalfaro/results/dbg_prunevid_capfix_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dbg_prunevid_capfix_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 0:30:00

# Smoke test for the PrunableDynamicCache kv_cache fix. Runs PruneVID at its
# standard cluster_ratio=0.5 (the master-table point) on a 300-sample slice
# only, to confirm the fix moves accuracy out of the ~38% collapse band
# before committing the full 3-point sweep re-run.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SRC_PRUNEVID=/project/rhu/dpalfaro/code/PruneVid

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/dbg_prunevid_capfix_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/prunevid-motionbenc/eval_prunevid_ov_capfix.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 --limit 8 \
    --prunevid --cluster_ratio 0.5 --temporal_segment_ratio 0.25 --selected_layer 10

echo "=== SMOKE RESULT ==="
python3 -c "
import json
s = json.load(open('${OUT}/summary.json'))
print('accuracy:', s['accuracy'])
print('correct:', s['correct'], '/', s['total_scoreable'])
"

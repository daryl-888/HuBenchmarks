#!/bin/bash
#SBATCH -J capfix_pv_r50
#SBATCH -o /project/rhu/dpalfaro/results/capfix_prunevid_r50_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/capfix_prunevid_r50_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# PruneVID @ cluster_ratio=0.5 on LLaVA-OV-7B, WITH the full fix
# (PrunableDynamicCache physical-prune + no-legacy-cache-roundtrip, PLUS the
# dycoke=True copy-paste bug removed from load_model -- that flag was
# silently wiping PruneVID's own pruning every decode step and triggering
# DyCoke's unrelated pruning path with stale state, which crashed).
# Replaces the pre-fix s1_prunevid_r50_run / w2_prunevid_ov_run point.

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
OUT=/project/rhu/dpalfaro/results/capfix_prunevid_r50_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/prunevid-motionbenc/eval_prunevid_ov_capfix.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --prunevid --cluster_ratio 0.5 --temporal_segment_ratio 0.25 --selected_layer 10

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method prunevid \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

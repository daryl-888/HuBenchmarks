#!/bin/bash
#SBATCH -J s1_prunevid_c100
#SBATCH -o /project/rhu/dpalfaro/results/s1_prunevid_c100_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_prunevid_c100_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 14:00:00

# PruneVID spatial-merge control: cluster_ratio=1.0 keeps one cluster per
# token, so the spatial merge is a no-op and only DPC-KNN temporal
# segmentation runs.
#
# Why: PruneVID-OV is flat at ~38% across cluster_ratio 0.10 / 0.25 / 0.50 on
# LLaVA-OV -- including 0.50, the authors' own default in scripts/eval.sh --
# so retention is not the variable. This isolates whether the temporal-segment
# stage alone collapses the backbone, or whether the spatial merge is needed
# to trigger it. It is the PruneVID analogue of the FastV keepall control,
# which returned the backbone and cleared the shared kv_cache path.
# See docs/RETENTION_DIAGNOSIS.md.

module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export SRC_PRUNEVID=/project/rhu/dpalfaro/code/PruneVid
export PYTHONPATH=${SRC_PRUNEVID}:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro
OUT=/project/rhu/dpalfaro/results/s1_prunevid_c100_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/prunevid-motionbenc/eval_prunevid_ov.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 --conv_template qwen_2 \
    --prunevid --cluster_ratio 1.0 --temporal_segment_ratio 0.25 --selected_layer 10

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method prunevid \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 || echo "GATE FAILED - do not record"

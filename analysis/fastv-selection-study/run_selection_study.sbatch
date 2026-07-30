#!/bin/bash
#SBATCH -J fv_sel
#SBATCH -o /project/rhu/dpalfaro/results/fv_sel_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/fv_sel_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 8:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu
#
# FastV selection study. Submit once per policy:
#   sbatch --export=SELECT=attention run_selection_study.sbatch
#   sbatch --export=SELECT=random    run_selection_study.sbatch
#   sbatch --export=SELECT=uniform   run_selection_study.sbatch
#
# Plus one CONTROL that isolates the harness from the policy:
#   sbatch --export=SELECT=attention,FASTV_R=0.00,TAG=keepall run_selection_study.sbatch
# r=0.00 keeps every visual token, so the run still goes through the exact same
# PrunableDynamicCache.kv_cache route but discards nothing. It must reproduce the
# 52.66% backbone. If it does not, the collapse belongs to the pruning harness
# that FastV and PruneVID-OV share, not to either method's selection rule.
#
# Everything except the layer-K selection policy is held fixed, and the budget
# (keep-count) is identical across all three: r=0.85 -> keep 15%.
#
# LIMIT=2000 covers 986 scoreable questions whose per-category mix matches the
# full 4,018 within ~1.5 pt, so subset accuracy is comparable to the full-run
# numbers. The `attention` arm is a CONTROL: it must reproduce w2_fastv_run
# restricted to idx < 2000, which is what proves this fork did not alter FastV.
#
# Env matches w2_run_fastv.sbatch exactly (dycoke11; the `fastv` env has broken
# torch). eager attention is forced on layer K only, inside load_model.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

SELECT="${SELECT:-attention}"
LIMIT="${LIMIT:-2000}"
FASTV_R="${FASTV_R:-0.85}"
# TAG lets the r=0.00 control write to its own directory instead of colliding
# with the r=0.85 arm that shares the same SELECT value.
OUT=/project/rhu/dpalfaro/results/fv_sel_${TAG:-$SELECT}

echo "=== FastV selection study: policy=${SELECT} r=${FASTV_R} limit=${LIMIT} ==="

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/analysis/fastv-selection-study/eval_fastv_selection.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir "${OUT}" \
    --num_frames 32 \
    --limit "${LIMIT}" \
    --fastv --fastv_k 2 --fastv_r "${FASTV_R}" \
    --select "${SELECT}" --seed 0

echo "=== done: ${OUT} ==="

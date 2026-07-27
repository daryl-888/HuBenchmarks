#!/bin/bash
#SBATCH -J ob_dyto_run
#SBATCH -o /project/rhu/dpalfaro/results/ob_dyto_run_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ob_dyto_run_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH -t 12:00:00

# GPU type is UNCONSTRAINED (--gpus=1, not ada:1) on purpose. Every ada GPU on
# this cluster is either allocated or in MIXED+PLANNED (reserved for a large
# pending job), so an ada-only request waits for a full drain regardless of
# priority — job 7833211 sat PENDING ~2 days at the top of the queue.
#
# DyTo loads Vicuna-7B in FP16 (eval_dyto.py:260), which V100 (compute 7.0,
# 32GB) supports natively — unlike the bf16 Qwen3-VL runs, which require ada.
# So this job can use volta, where capacity exists.
#
# CONSEQUENCE TO DISCLOSE: if this lands on volta it is the only result in the
# project produced on a different GPU architecture than the rest. Greedy decoding
# makes it deterministic on whatever card it gets, but it is not directly
# bit-comparable to the ada runs. The log records the GPU, so the record is
# unambiguous either way.

# DyTo full run on its own backbone (LLaVA-NeXT Vicuna-7B).
#
# THIS IS A LABELLED VARIANT, NOT DyTo-as-published. Two defects in the authors'
# released code had to be addressed:
#   1. finch_cluster() has no return statement -> RECOVERED by transcribing the
#      8-line tail verbatim from the sibling KMeans function in the same file
#      (13/14 normalized lines identical). Recovered code, not invented.
#   2. FINCH(tw_finch=...) is not a parameter of the pinned finch-clust==0.2.0 ->
#      RECONSTRUCTED from the published TW-FINCH rule (d_feat * |i-j|) via FINCH's
#      own initial_rank hook. This one is OUR implementation.
#
# Report only as "DyTo (reconstructed TW-FINCH)". summary.json carries
# finch_variant / paper_faithful:false / reconstruction_notes so the caveat
# travels with the data.
#
# Measured A/B (n=8, jobs 7786505 vs 7786513): TW-FINCH and standard FINCH gave
# IDENTICAL predictions, 0/8 differ. Do not claim the weighting changes results.
#
# 32 frames, not the paper's 100: 100 OOMs on a 44GiB card, and FINCH selects
# ~25 representative frames regardless.
module purge
module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE; unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/dyto_pkgs:/project/rhu/dpalfaro/DYTO:${PYTHONPATH:-}
export DYTO_FINCH_SHIM=1   # LABELLED VARIANT: reconstructed TW-FINCH (not authors' code)
# A/B control: DYTO_TW_OFF=1 falls back to standard FINCH, to prove the temporal
# weighting actually changes predictions rather than merely running without error.
export DYTO_TW_OFF=${DYTO_TW_OFF:-0}
export DYTO_ROOT=/project/rhu/dpalfaro/DYTO
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/rhu/dpalfaro

# Record the GPU this landed on — the result is annotated with it.
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
OUT=/project/rhu/dpalfaro/results/ob_dyto_run
/project/rhu/dpalfaro/conda/envs/dyto/bin/python3 \
    /project/rhu/dpalfaro/code/other-backbones/dyto-motionbenc/eval_dyto.py \
    --model-path /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --meta-path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output-dir ${OUT} \
    --num-frames 32
echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" --expect-method dyto || echo "GATE FAILED - do not record"

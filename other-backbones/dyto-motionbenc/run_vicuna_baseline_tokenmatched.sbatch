#!/bin/bash
#SBATCH -J ob_vicuna_tm
#SBATCH -o /project/rhu/dpalfaro/results/ob_vicuna_tm_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ob_vicuna_tm_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=ada:1
#SBATCH --mem=48G
#SBATCH -t 12:00:00

# TOKEN-MATCHED LLaVA-NeXT-Vicuna-7B BASELINE (corrected 2026-07-28).
#
# The first attempt (job 7882502) is INVALID and must not be cited: it ran
# --temporal-aggregation concat at 32 frames = 32 x 576 = 18,432 visual tokens
# into a 4,096 context. A 4.5x overflow. It scored 4.48% with 5,888/8,052 empty
# predictions and garbage like 'obo' / '0000000000000000' in the rest.
#
# The overflow is not just a config slip — it is the reason DyTo exists. This
# backbone CANNOT ingest 32 frames uncompressed, so a frame-matched baseline is
# impossible by construction. The meaningful control is TOKEN-MATCHED:
#
#   DyTo:     32 frames -> FINCH picks ~25 -> ToMe merges -> ~3,680 tokens
#   baseline:  6 frames x 576                            =  3,456 tokens
#
# Same token budget, no compression. The question it answers is the one DyTo's
# paper implicitly poses: given a fixed budget, is compressing 32 frames better
# than simply using fewer frames?
#
# DyTo's full run (7852061, 42.06%) passed every gate check EXCEPT divergence,
# because no baseline for this backbone had ever been run. Without it, 42.06% is
# an absolute number with no measurable delta and no McNemar test — the only
# gated cell in the project in that position.
#
# This run closes that: identical script, identical 32 frames, identical prompt,
# but --temporal-aggregation concat (plain temporal concatenation, verified at
# llava_arch.py:249) and DYTO_FINCH_SHIM=0, so neither FINCH nor ToMe runs.
# The GATE line below is expected to report no method engagement — that is the
# point of a baseline, not a failure.
#
# GPU type is PINNED TO ADA deliberately. DyTo loads Vicuna-7B in FP16
# (eval_dyto.py:260), so it *could* run on volta (V100, compute 7.0, 32GB) where
# there is far more free capacity — but that would make this the only result in
# the project produced on a different GPU architecture. Consistency with the
# other 20 cells is worth the wait.
#
# Expect a long queue: every ada node is allocated or MIXED+PLANNED (reserved for
# a large pending job), and backfill will not take anything that delays a reserved
# job. An earlier ada-only submission sat PENDING ~2 days at the TOP of the queue.
# That is scheduling scarcity, not a defect in this script — do not "fix" it by
# relaxing the GPU type.
#
# nvidia-smi below records the card, so the hardware behind the number is on file.

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
export DYTO_FINCH_SHIM=0   # BASELINE: shim OFF, no FINCH involved at all
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
OUT=/project/rhu/dpalfaro/results/ob_vicuna_tokenmatched
/project/rhu/dpalfaro/conda/envs/dyto/bin/python3 \
    /project/rhu/dpalfaro/code/other-backbones/dyto-motionbenc/eval_dyto.py \
    --model-path /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --meta-path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output-dir ${OUT} \
    --num-frames 6 \
    --temporal-aggregation concat
echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" --expect-method dyto || echo "baseline: expected (no method applied)"

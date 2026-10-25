#!/bin/bash
#SBATCH -J dyto_official_smoke
#SBATCH -o /project/rhu/dpalfaro/results/dyto_official_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dyto_official_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# =============================================================================
# DyTo — UNMODIFIED upstream verification. Smoke (8 samples).
#
# Runs DyTo's OWN run_inference.py -> run_inference_multiple_choice_qa.py on a
# pristine clone, NO patches. The verification EXPECTS a crash at the documented
# released-code defect (tw_finch / missing return). See README.md.
#
# ONE-TIME SETUP (login node):
#   1. bash /project/rhu/dpalfaro/code/dyto-official-verify/verify_upstream.sh
#   2. conda create -n dyto_v python=3.10 -y
#      /project/rhu/dpalfaro/conda/envs/dyto_v/bin/pip install \
#          torch==2.2.0 torchvision==0.17.0 \
#          --index-url https://download.pytorch.org/whl/cu121
#      /project/rhu/dpalfaro/conda/envs/dyto_v/bin/pip install \
#          -e /project/rhu/dpalfaro/code/dyto-official-verify/DYTO
#      /project/rhu/dpalfaro/conda/envs/dyto_v/bin/pip install finch-clust==0.2.0 decord
#   3. Weights exist: /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b
# =============================================================================

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYTO_PYTHON=/project/rhu/dpalfaro/conda/envs/dyto_v/bin/python3

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/dyto_v/bin/python3 \
    /project/rhu/dpalfaro/code/dyto-official-verify/eval_dyto_official.py \
    --meta         /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --video-base   /project/rhu/MotionBench_Data/MotionBench \
    --dyto-root    /project/rhu/dpalfaro/code/dyto-official-verify/DYTO \
    --model-path   /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --workdir      /project/rhu/dpalfaro/results/dyto_official_smoke \
    --num-frames   32 \
    --temporal-aggregation spatial_tome_finch_dynamic_all_frms \
    --rope-scaling 2 \
    --limit        8

# Verification runs are expected to FAIL (exit 0 = defect reproduced).
# A clean exit (0) here means the documented defect reproduced, which is a PASS.
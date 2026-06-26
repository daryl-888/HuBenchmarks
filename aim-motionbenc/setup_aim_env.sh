#!/bin/bash
# One-time AIM environment setup — run on Carya login node.
# AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning (ICCV 2025)
# https://github.com/LaVi-Lab/AIM
#
# Usage:
#   bash /project/rhu/dpalfaro/aim-motionbenc/setup_aim_env.sh

set -e

module purge
module load Miniforge3/py3.10

echo "=== Step 1: Clone AIM source ==="
if [ ! -d /project/rhu/dpalfaro/code/AIM ]; then
    cd /project/rhu/dpalfaro/code
    git clone https://github.com/LaVi-Lab/AIM
else
    echo "AIM already cloned at /project/rhu/dpalfaro/code/AIM, skipping."
fi

echo "=== Step 2: Create conda env ==="
if [ -d /project/rhu/dpalfaro/conda/envs/aim ]; then
    echo "aim env already exists, skipping."
else
    conda create -n aim python=3.10.14 -y
fi

PIP=/project/rhu/dpalfaro/conda/envs/aim/bin/pip

echo "=== Step 3: Install PyTorch 2.3.1 ==="
$PIP install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 4: Install AIM (main package) ==="
$PIP install -e "/project/rhu/dpalfaro/code/AIM[train]"

echo "=== Step 5: Install AIM's custom transformers ==="
$PIP install -e /project/rhu/dpalfaro/code/AIM/other_packages/transformers

echo "=== Step 6: Install AIM's custom lmms-eval ==="
$PIP install -e /project/rhu/dpalfaro/code/AIM/other_packages/lmms-eval

echo "=== Step 7: Install qwen-vl-utils ==="
$PIP install -e /project/rhu/dpalfaro/code/AIM/other_packages/qwen-vl-utils

echo "=== Step 8: Cache LLaVA-OV-Qwen2 model (if not already cached) ==="
# Weights already at /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 from FlashVID run
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
if [ ! -d /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 ]; then
    echo "Downloading llava-ov-7b-qwen2 weights..."
    /project/rhu/dpalfaro/conda/envs/dycoke11/bin/huggingface-cli download \
        lmms-lab/llava-onevision-qwen2-7b-ov \
        --local-dir /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2
else
    echo "llava-ov-7b-qwen2 already at /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2, skipping."
fi

echo ""
echo "=== AIM env setup complete ==="
echo "Tasks dir must be at: /project/rhu/dpalfaro/aim-motionbenc/tasks/"
echo "(SCP from local: scp -r aim-motionbenc/tasks/ dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/aim-motionbenc/)"
echo ""
echo "Submit smoke test with:"
echo "  sbatch /project/rhu/dpalfaro/aim-motionbenc/test_aim.sbatch"

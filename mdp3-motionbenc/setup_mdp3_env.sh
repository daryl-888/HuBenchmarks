#!/bin/bash
# One-time MDP3 environment setup — run on Carya login node.
# Takes ~15 min (conda clone is slow).
#
# Usage:
#   bash /project/rhu/dpalfaro/code/mdp3-motionbenc/setup_mdp3_env.sh

set -e

module purge
module load Miniforge3/py3.10

echo "=== Step 1: Clone MDP3 source ==="
if [ ! -d /project/rhu/dpalfaro/code/MDP3 ]; then
    cd /project/rhu/dpalfaro/code
    git clone https://github.com/sunh-23/MDP3
else
    echo "MDP3 already cloned, skipping."
fi

echo "=== Step 2: Create conda env (clone holitom) ==="
if [ -d /project/rhu/dpalfaro/conda/envs/mdp3 ]; then
    echo "mdp3 env already exists, skipping clone."
else
    conda create --name mdp3 --clone holitom -y
fi

PIP=/project/rhu/dpalfaro/conda/envs/mdp3/bin/pip
PYTHON=/project/rhu/dpalfaro/conda/envs/mdp3/bin/python3

echo "=== Step 3: Install MDP3 package ==="
$PIP install -e /project/rhu/dpalfaro/code/MDP3

echo "=== Step 4: Install extra deps ==="
$PIP install pysubs2

echo "=== Step 5: Reinstall CUDA-bundled torch (pip install MDP3 overwrites conda's torch) ==="
# MDP3's setup.py pulls in PyPI torch which lacks bundled CUDA libs → libcudart not found
# Force-reinstall from pytorch.org/whl/cu121 which bundles libcudart, libcublas, etc.
$PIP install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 6: Cache SigLip model (requires internet) ==="
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
$PYTHON -c "
from transformers import AutoModel, AutoProcessor
print('Downloading google/siglip-so400m-patch14-384 ...')
AutoModel.from_pretrained('google/siglip-so400m-patch14-384')
AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')
print('SigLip cached.')
"

echo ""
echo "=== MDP3 env setup complete ==="
echo "Submit the eval job with:"
echo "  sbatch /project/rhu/dpalfaro/code/mdp3-motionbenc/run_mdp3.sbatch"

#!/bin/bash
#SBATCH -J ovqwen2_mdp3_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/mdp3_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/mdp3_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# =============================================================================
# ONE-TIME SETUP (run on login node before submitting this job)
# =============================================================================
#
# 1. Clone MDP3:
#    cd /project/rhu/dpalfaro/code
#    git clone https://github.com/sunh-23/MDP3
#
# 2. Create conda env (clone holitom for compatible transformers==4.45.2):
#    conda create --name mdp3 --clone holitom
#
# 3. Install MDP3 and extras:
#    /project/rhu/dpalfaro/conda/envs/mdp3/bin/pip install -e /project/rhu/dpalfaro/code/MDP3
#    /project/rhu/dpalfaro/conda/envs/mdp3/bin/pip install pysubs2
#
# 3b. Install torch==2.3.1 + torchvision==0.18.1 to a dpalfaro-owned target dir.
#     The conda env (cloned from holitom) has torch 2.6.0; installing over it is
#     unreliable (EACCES or broken libcudart). Use --target to shadow it via PYTHONPATH.
#    mkdir -p /project/rhu/dpalfaro/mdp3_pkgs
#    /project/rhu/dpalfaro/conda/envs/mdp3/bin/pip install \
#        torch==2.3.1 torchvision==0.18.1 --no-deps \
#        --target /project/rhu/dpalfaro/mdp3_pkgs \
#        --index-url https://download.pytorch.org/whl/cu121
#
# 4. Cache SigLip (MDP3 frame selector uses it; must be cached before offline GPU job):
#    /project/rhu/dpalfaro/conda/envs/mdp3/bin/python3 -c "
#    from transformers import AutoModel, AutoProcessor
#    AutoModel.from_pretrained('google/siglip-so400m-patch14-384')
#    AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')
#    "
#
# 5. Sync this script to Carya:
#    scp -r mdp3-motionbenc/ dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/
#
# =============================================================================

module purge
module load Miniforge3/py3.10
module load cudatoolkit/12.1

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export WORLD_SIZE=1
# mdp3_pkgs: torch==2.3.1 + torchvision==0.18.1 installed via pip --target into a
# dpalfaro-owned dir. The conda env's torch is broken (2.6.0→2.3.1 --ignore-installed
# left libcudart.so.12 missing). mdp3_pkgs must be FIRST to shadow the conda env.
# MDP3 (vlmeval frame selector) must come before LLaVA-NeXT (llava model loader).
# LD_LIBRARY_PATH: torch's .so files (libcublas, libcudart) live in mdp3_pkgs/torch/lib/
# and need to be on the dynamic linker path.
export PYTHONPATH=/project/rhu/dpalfaro/mdp3_pkgs:/project/rhu/dpalfaro/code/MDP3:/project/rhu/dpalfaro/code/HoliTom/LLaVA-NeXT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/project/rhu/dpalfaro/mdp3_pkgs/torch/lib:/project/rhu/dpalfaro/conda/envs/mdp3/lib/python3.11/site-packages/nvidia/nccl/lib:/project/rhu/dpalfaro/conda/envs/mdp3/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/mdp3/bin/python3 \
    /project/rhu/dpalfaro/code/mdp3-motionbenc/eval_mdp3.py \
    --meta-file    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl --num_frames 32 \
    --video-roots  /project/rhu/MotionBench_Data/MotionBench \
    --model-path   /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 \
    --output-dir   /project/rhu/dpalfaro/results/mdp3_run1 \
    --conv-template qwen_2 \
    --pool-frames  32 \
    --select-frames 8 \
    --resume

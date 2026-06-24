#!/bin/bash
#SBATCH -J mdp3_motionbench
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
#    /project/rhu/dpalfaro/conda/envs/mdp3/bin/pip install torchvision pysubs2
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

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export WORLD_SIZE=1
# MDP3 (vlmeval frame selector) must come first; LLaVA-NeXT provides the llava model loader
export PYTHONPATH=/project/rhu/dpalfaro/code/MDP3:/project/rhu/dpalfaro/code/HoliTom/LLaVA-NeXT:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/mdp3/bin/python3 \
    /project/rhu/dpalfaro/code/mdp3-motionbenc/eval_mdp3.py \
    --meta-file    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --video-roots  /project/rhu/MotionBench_Data/MotionBench \
    --model-path   /project/rhu/dpalfaro/weights/llava-ov-7b \
    --output-dir   /project/rhu/dpalfaro/results/mdp3_run1 \
    --conv-template qwen_1_5 \
    --pool-frames  32 \
    --select-frames 8 \
    --resume

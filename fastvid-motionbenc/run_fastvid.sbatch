#!/bin/bash
#SBATCH -J fastvid_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/fastvid_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/fastvid_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
# FastVID's modified LLaVA-NeXT source must precede DyCoke's so its patched forward passes
# take effect. Verify the exact subpath after cloning:
#   ls /project/rhu/dpalfaro/code/FastVID/fastvid_llavaonevision/
# Adjust FASTVID_SRC below if the llava package lives one level deeper (e.g. .../LLaVA-NeXT).
FASTVID_SRC=/project/rhu/dpalfaro/code/FastVID/fastvid_llavaonevision
export PYTHONPATH=${FASTVID_SRC}:/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# FastVID hyperparameters — check FastVID/fastvid_llavaonevision/scripts/eval.sh to confirm
# whether these are read as env vars, model_args, or a config file inside their source.
export FASTVID_R=0.9         # token retention ratio (keep 90% of tokens)
export FASTVID_C=8           # number of temporal segments for DySeg
export FASTVID_TAU=0.9       # DySeg inter-frame similarity threshold
export FASTVID_DENSITY_D=0.1 # density pruning threshold

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/fastvid/bin/python3 -m lmms_eval \
    --include_path /project/rhu/dpalfaro/code/fastvid-motionbenc/tasks \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks motionbench \
    --batch_size 1 \
    --output_path /project/rhu/dpalfaro/results/fastvid_run1 \
    --log_samples

#!/bin/bash
#SBATCH -J dyto_test
#SBATCH -o /project/rhu/dpalfaro/results/dyto_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dyto_test_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# Smoke test — 50 samples. Complete setup steps in run_dyto.sbatch first.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export WORLD_SIZE=1
# DYTO/dyto must be on PYTHONPATH so builder.py's "from llava.model import *" resolves.
# Also need finch-clust installed in the conda env.
# dyto_pkgs: finch-clust installed via pip install --target (conda env is mahern69-owned)
export PYTHONPATH=/project/rhu/dpalfaro/dyto_pkgs:/project/rhu/dpalfaro/DYTO:/project/rhu/dpalfaro/DYTO/dyto:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/dyto/bin/python3 \
    /project/rhu/dpalfaro/dyto-motionbenc/eval_dyto.py \
    --model-path   /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --meta-path    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output-dir   /project/rhu/dpalfaro/results/dyto_test_${SLURM_JOB_ID} \
    --conv-template vicuna_v1 \
    --num-frames   100 \
    --temporal-aggregation spatial_tome_finch_dynamic_all_frms \
    --rope-scaling 2 \
    --limit 50

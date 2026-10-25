#!/bin/bash
#SBATCH -J dyto_official
#SBATCH -o /project/rhu/dpalfaro/results/dyto_official_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dyto_official_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# =============================================================================
# DyTo -- UNMODIFIED upstream verification. FULL 8,052-sample run.
#
# Expected: terminates at the first released-code defect (tw_finch or missing
# return); the crash IS the verification. If DyTo's own pipeline ever produces
# predictions, the wrapper scores them and README.md explains the meaning.
#
# ONE-TIME SETUP: see test_dyto_official.sbatch comments (verify_upstream.sh,
# dyto_v conda env, weights).
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
    --workdir      /project/rhu/dpalfaro/results/dyto_official \
    --num-frames   32 \
    --temporal-aggregation spatial_tome_finch_dynamic_all_frms \
    --rope-scaling 2
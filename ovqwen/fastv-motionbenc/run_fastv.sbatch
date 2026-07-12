#!/bin/bash
#SBATCH -J ovqwen15_fastv_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/fastv_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/fastv_%j.err
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
# FastV requires its custom transformers fork installed in the conda env.
# Clone FastV, then in the fastv env:
#   pip install -e /project/rhu/dpalfaro/code/FastV/src/FastV/llava-hf/transformers/
# After the Qwen2 port is complete, DyCoke's LLaVA-NeXT provides the llava package.
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/fastv/bin/python3 \
    /project/rhu/dpalfaro/code/fastv-motionbenc/eval_fastv.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/fastv_run1 \
    --num_frames 32 \
    --fastv \
    --fastv_k 2 \
    --fastv_r 0.5

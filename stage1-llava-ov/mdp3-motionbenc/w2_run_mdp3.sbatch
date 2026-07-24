#!/bin/bash
#SBATCH -J w2_mdp3
#SBATCH -o /project/rhu/dpalfaro/results/w2_mdp3_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/w2_mdp3_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# Smoke test — 50 samples. Complete setup steps in run_mdp3.sbatch first.

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
# LD_LIBRARY_PATH: torch's .so files (libcublas, libcudart) live in mdp3_pkgs/torch/lib/
# and need to be on the dynamic linker path.
export PYTHONPATH=/project/rhu/dpalfaro/mdp3_pkgs:/project/rhu/dpalfaro/code/MDP3:/project/rhu/dpalfaro/code/HoliTom/LLaVA-NeXT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/project/rhu/dpalfaro/conda/envs/mdp3/lib/python3.11/site-packages/nvidia/nccl/lib:/project/rhu/dpalfaro/mdp3_pkgs/torch/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/mdp3/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/mdp3-motionbenc/eval_mdp3.py \
    --meta-file    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --video-roots  /project/rhu/MotionBench_Data/MotionBench \
    --model-path   /project/rhu/dpalfaro/weights/llava-ov-7b \
    --output-dir /project/rhu/dpalfaro/results/w2_mdp3_run \
    --conv-template qwen_1_5 \
    --pool-frames  32 \
    --select-frames 8

OUTDIR=/project/rhu/dpalfaro/results/w2_mdp3_run
echo "=== GATE: did the method actually engage? ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUTDIR}" \
    --expect-method mdp3 || echo "GATE FAILED - do not record this run"

#!/bin/bash
#SBATCH -J mdp3_ovqwen2
#SBATCH -o /project/rhu/dpalfaro/results/mdp3_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/mdp3_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10
module load cudatoolkit/12.1

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export WORLD_SIZE=1
export PYTHONPATH=/project/rhu/dpalfaro/mdp3_pkgs:/project/rhu/dpalfaro/code/MDP3:/project/rhu/dpalfaro/code/HoliTom/LLaVA-NeXT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/project/rhu/dpalfaro/mdp3_pkgs/torch/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/mdp3/bin/python3 \
    /project/rhu/dpalfaro/code/ovqwen2/mdp3-motionbenc/eval_mdp3.py \
    --meta-file    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --num_frames   32 \
    --video-roots  /project/rhu/MotionBench_Data/MotionBench \
    --model-path   /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 \
    --output-dir   /project/rhu/dpalfaro/results/mdp3_ovqwen2_fresh \
    --conv-template qwen_2 \
    --pool-frames  32 \
    --select-frames 8

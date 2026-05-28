#!/bin/bash
# VideoITG Stage 1 test — 10 samples only
#SBATCH -J videoitg_test_grounding
#SBATCH -o /project/rhu/dpalfaro/results/videoitg_test_grounding_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/videoitg_test_grounding_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/VideoITG:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/videoitg/bin/python3 -m lmms_eval \
    --model videoitg \
    --model_args pretrained=/project/rhu/dpalfaro/weights/videoitg-8b,num_frames=512,target_fps=1,output_dir=/project/rhu/dpalfaro/results/videoitg_test_grounding \
    --include_path /project/rhu/dpalfaro/code/videoitg-motionbenc/tasks \
    --tasks motionbench \
    --batch_size 1 \
    --limit 10 \
    --output_path /project/rhu/dpalfaro/results/videoitg_test_grounding_lmms

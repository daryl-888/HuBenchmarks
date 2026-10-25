#!/bin/bash
#SBATCH -J ob_imove_smoke
#SBATCH -o /project/rhu/dpalfaro/results/ob_imove_smoke_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ob_imove_smoke_%j.err
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
export PYTHONPATH=/project/rhu/dpalfaro/code/iMOVE:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/imove/bin/python3 -m lmms_eval \
    --include_path /project/rhu/dpalfaro/code/imove-motionbenc/tasks \
    --model imove \
    --model_args pretrained=/project/rhu/dpalfaro/weights/imove \
    --tasks motionbench \
    --batch_size 1 \
    --limit 10 \
    --output_path /project/rhu/dpalfaro/results/imove_test \
    --log_samples

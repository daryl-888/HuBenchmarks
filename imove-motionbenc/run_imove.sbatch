#!/bin/bash
#SBATCH -J imove_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/imove_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/imove_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
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

# TODO: iMOVE is a trained model (not plug-and-play) — will need its own weights and
# possibly a different lmms_eval model backend. Check repo when code is released.
# iMOVE may need its own conda env with different dependencies.
/project/rhu/dpalfaro/conda/envs/imove/bin/python3 -m lmms_eval \
    --include_path /project/rhu/dpalfaro/code/imove-motionbenc/tasks \
    --model imove \
    --model_args pretrained=/project/rhu/dpalfaro/weights/imove \
    --tasks motionbench \
    --batch_size 1 \
    --output_path /project/rhu/dpalfaro/results/imove_run1 \
    --log_samples

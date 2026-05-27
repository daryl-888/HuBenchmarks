#!/bin/bash
#SBATCH -J dycoke_test
#SBATCH -o /project/rhu/dpalfaro/results/dycoke_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dycoke_test_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 -m lmms_eval \
    --include_path /project/rhu/dpalfaro/code/dycoke-motionbenc/tasks \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen,dycoke=True,dycoke_l=3,dycoke_p=0.8,dycoke_k=0.3 \
    --tasks motionbench \
    --batch_size 1 \
    --limit 10 \
    --output_path /project/rhu/dpalfaro/results/dycoke_test \
    --log_samples

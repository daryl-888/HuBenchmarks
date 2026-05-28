#!/bin/bash
#SBATCH -J holitom_test
#SBATCH -o /project/rhu/dpalfaro/results/holitom_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/holitom_test_%j.err
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
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro/code/HoliTom

WRAPPER=holitom \
RETAIN_RATIO=0.15 \
T=0.80 \
HOLITOM_k=18 \
HOLITOM_r=0.5 \
/project/rhu/dpalfaro/conda/envs/holitom/bin/python3 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
    --include_path /project/rhu/dpalfaro/code/holitom-motionbenc/tasks \
    --tasks motionbench \
    --batch_size 1 \
    --limit 10 \
    --log_samples \
    --log_samples_suffix holitom_test \
    --output_path /project/rhu/dpalfaro/results/holitom_test

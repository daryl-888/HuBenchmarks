#!/bin/bash
#SBATCH -J sttm_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/sttm_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/sttm_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
# STTM first (quadtree patch), then DyCoke (lmms_eval + NFS-fixed llava_onevision)
export PYTHONPATH=/project/rhu/dpalfaro/code/STTM:/project/rhu/dpalfaro/code/DyCoke:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/sttm/bin/python3 \
    /project/rhu/dpalfaro/code/sttm-motionbenc/run_sttm_eval.py \
    --include_path /project/rhu/dpalfaro/code/sttm-motionbenc/tasks \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks motionbench \
    --batch_size 1 \
    --output_path /project/rhu/dpalfaro/results/sttm_run1 \
    --log_samples

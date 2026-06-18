#!/bin/bash
#SBATCH -J fastvid_test
#SBATCH -o /project/rhu/dpalfaro/results/fastvid_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/fastvid_test_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
FASTVID_SRC=/project/rhu/dpalfaro/code/FastVID/fastvid_llavaonevision
export PYTHONPATH=${FASTVID_SRC}:/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export FASTVID_R=0.9
export FASTVID_C=8
export FASTVID_TAU=0.9
export FASTVID_DENSITY_D=0.1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/fastvid/bin/python3 -m lmms_eval \
    --include_path /project/rhu/dpalfaro/code/fastvid-motionbenc/tasks \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks motionbench \
    --batch_size 1 \
    --output_path /project/rhu/dpalfaro/results/fastvid_test_${SLURM_JOB_ID} \
    --log_samples \
    --limit 50

#!/bin/bash
#SBATCH -J aim_test
#SBATCH -o /project/rhu/dpalfaro/results/aim_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/aim_test_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

# ~/.local packages (torch 2.3.1, torchvision 0.18.1) must come before conda env's
# older torchvision to avoid register_fake AttributeError
export PYTHONPATH=/home/dpalfaro/.local/lib/python3.11/site-packages:/project/rhu/dpalfaro/AIM/llava:/project/rhu/dpalfaro/AIM/other_packages/transformers/src:/project/rhu/dpalfaro/AIM/other_packages/lmms-eval:/project/rhu/dpalfaro/AIM:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

# Smoke test: 50 samples
/project/rhu/dpalfaro/conda/envs/aim/bin/accelerate launch \
    --num_processes=1 --gpu_ids 0 --main_process_port 25666 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=eager \
    --tasks motionbench \
    --batch_size 1 \
    --limit 50 \
    --log_samples \
    --log_samples_suffix aim_test \
    --output_path /project/rhu/dpalfaro/results/aim_test_${SLURM_JOB_ID} \
    --include_path /project/rhu/dpalfaro/aim-motionbenc/tasks
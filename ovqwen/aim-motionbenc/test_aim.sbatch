#!/bin/bash
#SBATCH -J ovqwen15_aim_motionbench
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

# aim_pkgs: dpalfaro-owned dir with torch==2.3.1 + torchvision==0.18.1 installed via
# pip install --target (conda env is owned by mahern69 — can't force-reinstall into it).
# Must come FIRST so these shadow the conda env's incompatible torch/torchvision.
export PYTHONPATH=/project/rhu/dpalfaro/aim_pkgs:/project/rhu/dpalfaro/AIM/llava:/project/rhu/dpalfaro/AIM/other_packages/transformers/src:/project/rhu/dpalfaro/AIM/other_packages/lmms-eval:/project/rhu/dpalfaro/AIM:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

# Smoke test: 50 samples
# Use python -m accelerate.commands.launch instead of the accelerate binary.
# The `accelerate` CLI binary imports accelerate_cli.py which eagerly imports estimate.py
# which imports timm which imports torchvision against the conda env's old torch →
# AttributeError: module 'torch.library' has no attribute 'register_fake'.
# python -m accelerate.commands.launch bypasses that import chain entirely.
/project/rhu/dpalfaro/conda/envs/aim/bin/python3 -m accelerate.commands.launch \
    --num_processes=1 --gpu_ids 0 --main_process_port 25666 \
    -m lmms_eval \
    --model llava_onevision \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b --conv_template qwen_1_5 --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl --num_frames 32 \ --output_dir /project/rhu/dpalfaro/results/ovqwen_aim_run1
     \
     \
    --limit 50 \
     \
    _suffix aim_test \
    --output_path /project/rhu/dpalfaro/results/aim_test_${SLURM_JOB_ID} \
    
#!/bin/bash
#SBATCH -J aim_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/aim_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/aim_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

# aim_pkgs: dpalfaro-owned dir with torch==2.3.1 + torchvision==0.18.1 installed via
# pip install --target (conda env is owned by mahern69 — can't force-reinstall into it).
# Must come FIRST so these shadow the conda env's incompatible torch/torchvision.
# AIM/llava — token merging in llava_arch.py
# AIM/other_packages/transformers/src — patched Qwen2 with token pruning
# AIM/other_packages/lmms-eval — patched lmms_eval with AIM configurations
export PYTHONPATH=/project/rhu/dpalfaro/aim_pkgs:/project/rhu/dpalfaro/AIM/llava:/project/rhu/dpalfaro/AIM/other_packages/transformers/src:/project/rhu/dpalfaro/AIM/other_packages/lmms-eval:/project/rhu/dpalfaro/AIM:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

# AIM with LLaVA-OV-Qwen2 backbone (same as FlashVID)
# Backbone: LLaVA-OneVision-7B (Qwen2 LLM, not Qwen1.5)
# Weights: /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2
# attn_implementation=eager enables token pruning (reads attention weights)
# merge_ratio=0.125: retain ~12.5% of visual tokens after merging (default paper: 25%)
# l1=14, l2=22: pruning scheduler params (from paper Table 5, Exp 6)
# Use python -m accelerate.commands.launch (not the accelerate binary) to avoid
# the accelerate_cli.py → estimate.py → timm → torchvision register_fake crash.
/project/rhu/dpalfaro/conda/envs/aim/bin/python3 -m accelerate.commands.launch \
    --num_processes=1 --gpu_ids 0 --main_process_port 25666 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=eager \
    --tasks motionbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix aim_motionbench \
    --output_path /project/rhu/dpalfaro/results/aim_run1 \
    --include_path /project/rhu/dpalfaro/aim-motionbenc/tasks
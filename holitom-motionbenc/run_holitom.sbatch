#!/bin/bash
#SBATCH -J holitom_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/holitom_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/holitom_%j.err
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
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

# HoliTom patches lmms_eval's llava_onevision.py — run from HoliTom repo root
# so its modified llava_onevision.py takes precedence over stock lmms_eval.
cd /project/rhu/dpalfaro/code/HoliTom

# HoliTom env vars (recommended values for LLaVA-OV-7B from HoliTom paper Table 4):
#   WRAPPER=holitom     — activates HoliTom model wrapping
#   RETAIN_RATIO=0.15   — keep 15% of visual tokens (outer-LLM stage)
#   T=0.80              — temporal segmentation threshold
#   HOLITOM_k=18        — inner-LLM: number of token clusters per layer
#   HOLITOM_r=0.5       — inner-LLM: merge ratio within each cluster
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
    --log_samples \
    --log_samples_suffix holitom \
    --output_path /project/rhu/dpalfaro/results/holitom_run1

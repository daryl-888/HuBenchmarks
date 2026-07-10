#!/bin/bash
#SBATCH -J ovqwen2_holitom_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/holitom_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/holitom_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
# HoliTom/LLaVA-NeXT must come first — it ships a patched siglip_encoder.py
# that returns attention weights when WRAPPER=holitom is set.
export PYTHONPATH=/project/rhu/dpalfaro/code/HoliTom/LLaVA-NeXT:/project/rhu/dpalfaro/code/HoliTom:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

# HoliTom env vars — must be set before python starts (siglip_encoder reads WRAPPER at import)
WRAPPER=holitom \
RETAIN_RATIO=0.15 \
T=0.80 \
HOLITOM_k=18 \
HOLITOM_r=0.5 \
/project/rhu/dpalfaro/conda/envs/holitom/bin/python3 \
    /project/rhu/dpalfaro/code/ovqwen2/holitom-motionbenc/eval_holitom.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/holitom_run1 \
    --num_frames 32

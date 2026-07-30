#!/bin/bash
#SBATCH -J samp_holitom
#SBATCH -o /project/rhu/dpalfaro/results/samp_holitom_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/samp_holitom_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00

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
    /project/rhu/dpalfaro/code/sampled-llava-ov/eval_holitom_sampled.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/samp_holitom_run \
    --num_frames 32 \
    --temperature 0.7 --top_p 0.9 --seed 0

OUT=/project/rhu/dpalfaro/results/samp_holitom_run
# NO GATE: sampling breaks the divergence check (two sampled runs differ by
# chance), so --vs-baseline cannot prove engagement. Distribution analysis is
# done afterwards by sampled-llava-ov/build_distribution.py.

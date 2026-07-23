#!/bin/bash
#SBATCH -J holitom_ovqwen2
#SBATCH -o /project/rhu/dpalfaro/results/holitom_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/holitom_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

## HoliTom × OVQwen 2
## Note: eval_holitom.py hardcodes conv_template internally — no --conv_template argument

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/project/rhu/dpalfaro/code/HoliTom/Llava-NeXT:/project/rhu/dpalfaro/code/HoliTom
export WRAPPER=holitom
export RETAIN_RATIO=0.15
export T=0.80
export HOLITOM_k=18
export HOLITOM_r=0.5

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/holitom/bin/python3 \
    /project/rhu/dpalfaro/code/ovqwen2/holitom-motionbenc/eval_holitom.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2 \
    --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/holitom_ovqwen2_fresh \
    --num_frames 32

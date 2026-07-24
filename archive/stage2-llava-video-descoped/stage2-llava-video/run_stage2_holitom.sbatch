#!/bin/bash
#SBATCH -J s2_run_stage2_holitom.sbatch_run_stage2_holitom
#SBATCH -o /project/rhu/dpalfaro/results/s2_run_stage2_holitom.sbatch_run_stage2_holitom_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s2_run_stage2_holitom.sbatch_run_stage2_holitom_%j.err
#SBATCH -N 1 -n 8 --gpus=ada:1 --mem=64G -t 48:00:00
#SBATCH --mail-type=END --mail-user=dpalfaro@cougarnet.uh.edu
module purge && module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1 HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:/project/rhu/dpalfaro/code/HoliTom
export WRAPPER=holitom RETAIN_RATIO=0.15 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5
cd /project/rhu/dpalfaro
/project/rhu/dpalfaro/conda/envs/holitom/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/holitom-motionbenc/eval_holitom.py \
    --model_path /project/rhu/dpalfaro/weights/llava-video-7b \
    --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/stage2_holitom_baseline --num_frames 32

#!/bin/bash
#SBATCH -J stage2_sttm -o /project/rhu/dpalfaro/results/stage2_sttm_%j.out -e /project/rhu/dpalfaro/results/stage2_sttm_%j.err
#SBATCH -N 1 -n 8 --gpus=ada:1 --mem=64G -t 48:00:00
#SBATCH --mail-type=END --mail-user=dpalfaro@cougarnet.uh.edu
module purge && module load Miniforge3/py3.10
export PYTHONNOUSERSITE=1 HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/project/rhu/dpalfaro/code/STTM:${PYTHONPATH:-}
cd /project/rhu/dpalfaro
/project/rhu/dpalfaro/conda/envs/sttm_new/bin/python3 \
    /project/rhu/dpalfaro/code/stage2-llava-video/sttm-llavavid/eval_sttm_llavavid.py \
    --model_path /project/rhu/dpalfaro/weights/llava-video-7b \
    --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/stage2_sttm_baseline --num_frames 32

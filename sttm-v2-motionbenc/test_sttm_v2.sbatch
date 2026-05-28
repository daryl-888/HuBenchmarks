#!/bin/bash
#SBATCH -J sttm_v2_test
#SBATCH -o /project/rhu/dpalfaro/results/sttm_v2_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/sttm_v2_test_%j.err
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
# STTM only — no DyCoke needed (no lmms_eval)
export PYTHONPATH=/project/rhu/dpalfaro/code/STTM:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/sttm/bin/python3 \
    /project/rhu/dpalfaro/code/sttm-v2-motionbenc/eval_sttm.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/sttm_v2_test_${SLURM_JOB_ID} \
    --num_frames 32 \
    --limit 10 \
    --sa_start_layer_idx     2 \
    --sa_tree_thresh         0.85 \
    --sa_tree_temporal_thresh 0.65 \
    --sa_tree_root_level     1

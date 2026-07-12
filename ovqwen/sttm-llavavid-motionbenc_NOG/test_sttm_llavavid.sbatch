#!/bin/bash
#SBATCH -J ovqwen15_sttm-llavavid_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/sttm_llavavid_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/sttm_llavavid_test_%j.err
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
export PYTHONPATH=/project/rhu/dpalfaro/code/STTM:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/sttm_new/bin/python3 \
    /project/rhu/dpalfaro/code/sttm-llavavid-motionbenc/eval_sttm_llavavid.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/sttm_llavavid_test_${SLURM_JOB_ID} \
    --num_frames 32 \
    --limit 50 \
    --sa_start_layer_idx      2 \
    --sa_tree_thresh          0.85 \
    --sa_tree_temporal_thresh 0.65 \
    --sa_tree_root_level      1

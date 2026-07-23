#!/bin/bash
#SBATCH -J ovqwen2_fastvid_test
#SBATCH -o /project/rhu/dpalfaro/results/fastvid_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/fastvid_test_%j.err
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
export PYTHONPATH=/project/rhu/dpalfaro/code/FastVID/fastvid_llavaonevision/LLaVA-NeXT:/project/rhu/dpalfaro/code/FastVID/fastvid_llavaonevision/lmms-eval:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/fastvid/bin/python3 -m lmms_eval \
     \
    --model llava_onevision \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b --conv_template qwen_2 --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl --num_frames 32 \ --output_dir /project/rhu/dpalfaro/results/stage1_fastvid_run1
     \
     \
    --output_path /project/rhu/dpalfaro/results/fastvid_test_${SLURM_JOB_ID} \
     \
    --limit 50

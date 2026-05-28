#!/bin/bash
# VideoITG Stage 2 test — 10 samples only. Run after test_grounding.sbatch.
#SBATCH -J videoitg_test
#SBATCH -o /project/rhu/dpalfaro/results/videoitg_test_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/videoitg_test_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/VideoITG:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

GROUNDING_JSONL=/project/rhu/dpalfaro/results/videoitg_test_grounding/motionbench_frame_scores.jsonl

/project/rhu/dpalfaro/conda/envs/videoitg/bin/python3 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen,frame_indices_jsonl=${GROUNDING_JSONL},num_frame=32 \
    --include_path /project/rhu/dpalfaro/code/videoitg-motionbenc/tasks \
    --tasks motionbench \
    --batch_size 1 \
    --limit 10 \
    --log_samples \
    --log_samples_suffix videoitg_test \
    --output_path /project/rhu/dpalfaro/results/videoitg_test

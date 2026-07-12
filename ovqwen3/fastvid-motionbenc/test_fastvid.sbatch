#!/bin/bash
#SBATCH -J fastvid_test
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
    --model_args pretrained=/project/rhu/dpalfaro/weights/llava-ov-7b,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32,attn_implementation=sdpa,fastvid_retention_ratio=0.10,fastvid_DySeg_c=8,fastvid_DySeg_tau=0.9,fastvid_STPrune_d=0.4,fastvid_DTM_p=4,fastvid_DTM_beta=0.6 \
     \
     \
    --output_path /project/rhu/dpalfaro/results/fastvid_test_${SLURM_JOB_ID} \
     \
    --limit 50

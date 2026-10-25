#!/bin/bash
#SBATCH -J s1_holitom_smoke2
#SBATCH -o /project/rhu/dpalfaro/results/s1_holitom_smoke2_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_holitom_smoke2_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/HoliTom/LLaVA-NeXT:/project/rhu/dpalfaro/code/HoliTom:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

WRAPPER=holitom \
RETAIN_RATIO=0.15 \
T=0.80 \
HOLITOM_k=18 \
HOLITOM_r=0.5 \
/project/rhu/dpalfaro/conda/envs/holitom/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/holitom-motionbenc/eval_holitom.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/holitom_test_${SLURM_JOB_ID} \
    --num_frames 32 --conv_template qwen_2 \
    --limit 50

OUTDIR=/project/rhu/dpalfaro/results/holitom_test_${SLURM_JOB_ID}

echo "=== GATE: did the method actually engage? ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUTDIR}" \
    --expect-method holitom \
    --vs-baseline /project/rhu/dpalfaro/results/fastv_run1 --smoke || echo "GATE FAILED - do not record this run"

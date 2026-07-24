#!/bin/bash
#SBATCH -J s1_fastv_smoke
#SBATCH -o /project/rhu/dpalfaro/results/s1_fastv_smoke_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s1_fastv_smoke_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 0:30:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# FastV smoke test — verifies the paper-exact port actually engages.
# Uses dycoke11 (known-good env; the `fastv` env has broken torch).
# eager attention is forced inside eval_fastv.load_model when --fastv is set.

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

OUT=/project/rhu/dpalfaro/results/fastv_smoke_${SLURM_JOB_ID}

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/stage1-llava-ov/fastv-motionbenc/eval_fastv.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --fastv --fastv_k 2 --fastv_r 0.85 \
    --limit 8

echo "=== GATE: did FastV actually prune (predictions differ from baseline)? ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method fastv || echo "(gate: single-run checks; --vs-baseline needs a matching baseline run)"

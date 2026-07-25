#!/bin/bash
#SBATCH -J s3_prunevid_smoke
#SBATCH -o /project/rhu/dpalfaro/results/s3_prunevid_smoke_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_prunevid_smoke_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 0:12:00

# PruneVID ported to Qwen3-VL — the last portable cell of the 22-cell matrix.
# Must print "PruneVID(Qwen3-VL) ACTIVE: ..." AND diverge from the baseline.
# ACTIVE alone is not enough: this method's LLaVA-OV port printed ACTIVE while
# producing byte-identical output, and only the divergence gate caught it.
#
# Env qwen3vl is the only one that can load Qwen3VLForConditionalGeneration.
# PYTHONPATH must reach the PruneVID checkout for models.pllava.modeling_pllava
# (the authors' cluster_dpc_knn).

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SRC_PRUNEVID=/project/rhu/dpalfaro/code/PruneVid
export PYTHONPATH=${SRC_PRUNEVID}:${PYTHONPATH}

cd /project/rhu/dpalfaro

OUT=/project/rhu/dpalfaro/results/s3_prunevid_smoke_${SLURM_JOB_ID}

/project/rhu/dpalfaro/conda/envs/qwen3vl/bin/python3 \
    /project/rhu/dpalfaro/code/stage3-qwen3-vl/prunevid-motionbenc/eval_prunevid_qwen3vl.py \
    --model_path /project/rhu/dpalfaro/weights/qwen3-vl-8b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir ${OUT} \
    --num_frames 32 \
    --prunevid --cluster_ratio 0.5 --temporal_segment_ratio 0.25 --selected_layer 10 \
    --limit 8

echo "=== GATE ==="
python3 /project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py "${OUT}" \
    --expect-method prunevid \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1 --smoke || echo "GATE FAILED - do not record"

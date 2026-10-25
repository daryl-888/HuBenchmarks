#!/bin/bash
#SBATCH -J samp_baseline
#SBATCH -o /project/rhu/dpalfaro/results/samp_baseline_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/samp_baseline_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# FastV smoke test — verifies the paper-exact port actually engages.
# Uses dycoke11 (known-good env; the `fastv` env has broken torch).
# eager attention is forced inside eval_fastv.load_model when is set.

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

OUT=/project/rhu/dpalfaro/results/samp_baseline_run

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 \
    /project/rhu/dpalfaro/code/sampled-llava-ov/eval_baseline_sampled.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/samp_baseline_run \
    --num_frames 32 \
    --temperature 0.7 --top_p 0.9 --seed 0

# NO GATE: sampling breaks the divergence check (two sampled runs differ by
# chance), so --vs-baseline cannot prove engagement. Distribution analysis is
# done afterwards by sampled-llava-ov/build_distribution.py.
    --expect-method fastv || echo "(gate: single-run checks; --vs-baseline needs a matching baseline run)"

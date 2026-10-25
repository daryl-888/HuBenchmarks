#!/bin/bash
#SBATCH -J ob_prunevid_run_cache_dataset
#SBATCH -o /project/rhu/dpalfaro/results/ob_prunevid_run_cache_dataset_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ob_prunevid_run_cache_dataset_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 -c "
from datasets import load_dataset
print('Loading and caching dataset...')
ds = load_dataset(
    'json',
    data_files='/project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl',
    cache_dir='/project/rhu/dpalfaro/cache/huggingface/datasets'
)
print('Done caching:', len(ds['train']), 'samples')
"

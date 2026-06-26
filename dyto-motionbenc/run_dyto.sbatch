#!/bin/bash
#SBATCH -J dyto_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/dyto_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/dyto_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

# DyTo: Dynamic Token Merging for Zero-Shot Video Understanding (ICCV 2025)
# Backbone: LLaVA-NeXT Vicuna-7B (Llama-2 LLM, CLIP encoder)
# temporal_aggregation=spatial_tome_finch_dynamic_all_frms:
#   - FINCH clusters 100 input frames → ~25 representative frames
#   - ToMe merges tokens per frame with dynamic ratio → ~3,680 total tokens
#
# =============================================================================
# ONE-TIME SETUP (run on login node before submitting this job)
# =============================================================================
#
# 1. Clone DyTo to a path YOU own (not /project/rhu/dpalfaro/code/ — shared with mahern69):
#    git clone https://github.com/Jam1ezhang/DYTO /project/rhu/dpalfaro/DYTO
#
# 2. Patch DyTo's __init__ (LlavaLlamaForCausalLM import fails without this):
#    python3 - <<'EOF'
#    TARGET = "/project/rhu/dpalfaro/DYTO/dyto/llava/__init__.py"
#    OLD = "from .model import LlavaLlamaForCausalLM"
#    NEW = "try:\n    from .model import LlavaLlamaForCausalLM\nexcept ImportError:\n    pass"
#    with open(TARGET) as f: content = f.read()
#    with open(TARGET, "w") as f: f.write(content.replace(OLD, NEW, 1))
#    print("Patched.")
#    EOF
#
# 3. Download LLaVA-NeXT Vicuna-7B weights:
#    export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
#    /project/rhu/dpalfaro/conda/envs/dycoke11/bin/huggingface-cli download \
#        liuhaotian/llava-v1.6-vicuna-7b \
#        --local-dir /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b
#
# 4. Create conda env (fresh — DyTo pins torch==2.2.0, transformers==4.38.2):
#    conda create -n dyto python=3.10 -y
#    /project/rhu/dpalfaro/conda/envs/dyto/bin/pip install \
#        torch==2.2.0 torchvision==0.17.0 \
#        --index-url https://download.pytorch.org/whl/cu121
#    /project/rhu/dpalfaro/conda/envs/dyto/bin/pip install \
#        -e /project/rhu/dpalfaro/DYTO
#    /project/rhu/dpalfaro/conda/envs/dyto/bin/pip install \
#        finch-clust==0.2.0 decord
#
# 5. Sync eval script to Carya:
#    scp -r dyto-motionbenc/ dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/
#
# =============================================================================

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export WORLD_SIZE=1
# DyTo provides its own dyto.llava package — do NOT add HoliTom/LLaVA-NeXT to PYTHONPATH
export PYTHONPATH=/project/rhu/dpalfaro/DYTO:/project/rhu/dpalfaro/DYTO/dyto:${PYTHONPATH:-}
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/dyto/bin/python3 \
    /project/rhu/dpalfaro/code/dyto-motionbenc/eval_dyto.py \
    --model-path   /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \
    --meta-path    /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output-dir   /project/rhu/dpalfaro/results/dyto_run1 \
    --conv-template image_seq_v3 \
    --num-frames   100 \
    --temporal-aggregation spatial_tome_finch_dynamic_all_frms \
    --rope-scaling 2 \
    --resume

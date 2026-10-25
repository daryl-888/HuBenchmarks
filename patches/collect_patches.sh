#!/bin/bash
# Run this on Carya to populate the patches/ directory from the live fixed files.
# Then scp the patches/ dir back to this repo.
#
# Usage (on Carya):
#   cd /project/rhu/dpalfaro/code/holitom-motionbenc
#   bash patches/collect_patches.sh

set -e

HOLITOM=/project/rhu/dpalfaro/code/HoliTom
PRUNEVID=/project/rhu/dpalfaro/code/PruneVid
DEST=$(dirname "$0")

# HoliTom patches are already in the repo (committed 2026-06-11).
# Uncomment these lines only if you need to refresh them from Carya.
#
# echo "Collecting HoliTom patches..."
# cp "$HOLITOM/holitom/modeling_qwen2.py"                                \
#    "$DEST/holitom/modeling_qwen2.py"
# cp "$HOLITOM/LLaVA-NeXT/llava/model/builder.py"                       \
#    "$DEST/holitom/builder.py"
# cp "$HOLITOM/LLaVA-NeXT/llava/model/multimodal_encoder/siglip_encoder.py" \
#    "$DEST/holitom/siglip_encoder.py"
# cp "$HOLITOM/LLaVA-NeXT/llava/__init__.py"                            \
#    "$DEST/holitom/llava_init.py"
# TQWEN=/project/rhu/dpalfaro/conda/envs/holitom/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py
# [ -f "$TQWEN" ] && cp "$TQWEN" "$DEST/holitom/transformers_qwen2.py"

echo "Collecting PruneVid patches (still missing from repo)..."
cp "$PRUNEVID/models/pllava/llama.py"           "$DEST/prunevid/llama.py"
cp "$PRUNEVID/models/pllava/modeling_pllava.py" "$DEST/prunevid/modeling_pllava.py"

echo "Done. Now on your local machine:"
echo "  scp -r dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/holitom-motionbenc/patches/ ."
echo "  git add patches/ && git commit -m 'patches: add Carya-fixed source files'"

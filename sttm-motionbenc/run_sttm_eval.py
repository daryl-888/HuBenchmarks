#!/usr/bin/env python3
"""
Wrapper that applies STTM's quadtree attention patch to Qwen2Model BEFORE
lmms_eval loads the LLaVA-OV-7B model, then hands off to lmms_eval's CLI.

Usage: called by run_sttm.sbatch / test_sttm.sbatch with the same args
as a normal lmms_eval invocation.

STTM patches Qwen2Model.forward() in-place on the class object, so any
subsequent instantiation uses quadtree attention automatically.
No pre-extraction of features needed — vision encoding runs normally.
"""
import runpy
import sys

from token_merging_monkey_patch.quadtree_attn_monkey_patch import (
    replace_qwen2_with_quadtree_attn,
)

SA_START_LAYER_IDX = 2
SA_TREE_THRESH = 0.85
SA_TREE_TEMPORAL_THRESH = 0.65
SA_TREE_ROOT_LEVEL = 1

replace_qwen2_with_quadtree_attn(
    sa_start_layer_idx=SA_START_LAYER_IDX,
    sa_tree_thresh=SA_TREE_THRESH,
    sa_tree_temporal_thresh=SA_TREE_TEMPORAL_THRESH,
    sa_tree_root_level=SA_TREE_ROOT_LEVEL,
)
print(
    f"STTM patch applied: layer={SA_START_LAYER_IDX} "
    f"thresh={SA_TREE_THRESH} temporal_thresh={SA_TREE_TEMPORAL_THRESH} "
    f"root_level={SA_TREE_ROOT_LEVEL}",
    flush=True,
)

runpy.run_module("lmms_eval", run_name="__main__", alter_sys=True)

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

# LlavaQwenConfig lacks max_batch_size, which STTM's patched Qwen2Model.forward()
# accesses at inference time. Inject it right after load_pretrained_model returns.
import llava.model.builder as _builder
_orig_load = _builder.load_pretrained_model
def _patched_load(*args, **kwargs):
    result = _orig_load(*args, **kwargs)
    _tok, model, _ip, _ctx = result
    if not hasattr(model.config, 'max_batch_size'):
        model.config.max_batch_size = 32
    return result
_builder.load_pretrained_model = _patched_load

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

# STTM's patch was written against an older transformers that didn't have
# num_logits_to_keep. Wrap the patched forward to accept and discard it.
import inspect
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as _Qwen2CausalLM
_sttm_forward = _Qwen2CausalLM.forward
if 'num_logits_to_keep' not in inspect.signature(_sttm_forward).parameters:
    def _compat_forward(*args, num_logits_to_keep=0, **kwargs):
        return _sttm_forward(*args, **kwargs)
    _Qwen2CausalLM.forward = _compat_forward

print(
    f"STTM patch applied: layer={SA_START_LAYER_IDX} "
    f"thresh={SA_TREE_THRESH} temporal_thresh={SA_TREE_TEMPORAL_THRESH} "
    f"root_level={SA_TREE_ROOT_LEVEL}",
    flush=True,
)

runpy.run_module("lmms_eval", run_name="__main__", alter_sys=True)

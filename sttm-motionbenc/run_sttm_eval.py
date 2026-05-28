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

# transformers 4.40 Qwen2Model lacks _update_causal_mask, which STTM's
# Qwen2Model_forward calls. Add a compatible implementation before the patch runs.
import torch as _torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as _Qwen2Model
if not hasattr(_Qwen2Model, '_update_causal_mask'):
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position=None,
                             past_key_values=None, output_attentions=False):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = _torch.finfo(dtype).min
        seq_len = input_tensor.shape[1]
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = _torch.arange(past_seen, past_seen + seq_len, device=device)
        target_len = (
            attention_mask.shape[-1] if isinstance(attention_mask, _torch.Tensor)
            else past_seen + seq_len + 1
        )
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        mask = _torch.full((seq_len, target_len), fill_value=min_dtype, dtype=dtype, device=device)
        if seq_len != 1:
            mask = _torch.triu(mask, diagonal=1)
        mask *= _torch.arange(target_len, device=device) > cache_position.reshape(-1, 1)
        mask = mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            mask = mask.clone()
            pad = mask[:, :, :, :attention_mask.shape[-1]] + attention_mask[:, None, None, :]
            mask[:, :, :, :attention_mask.shape[-1]] = mask[:, :, :, :attention_mask.shape[-1]].masked_fill(
                pad == 0, min_dtype
            )
        return mask
    _Qwen2Model._update_causal_mask = _update_causal_mask
    print("Added _update_causal_mask to Qwen2Model", flush=True)

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

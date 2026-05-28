#!/usr/bin/env python3
"""
Wrapper that applies STTM's quadtree attention patch to Qwen2Model BEFORE
lmms_eval loads the LLaVA-OV-7B model, then hands off to lmms_eval's CLI.

Compatibility shims for transformers 4.40 / 4.45 vs STTM's 4.40.0.dev0 API:
  1. max_batch_size      — LlavaQwenConfig lacks it; inject after model load
  2. _update_causal_mask — missing in transformers 4.40 Qwen2Model; add if absent
  3. image_token_start_index / image_token_length / num_frame
                         — STTM-specific per-sample attrs; set via hook on
                           prepare_inputs_labels_for_multimodal
  4. num_logits_to_keep  — added to generate() in 4.40 stable; STTM's replaced
                           Qwen2ForCausalLM.forward doesn't accept it; shim it
"""
import runpy
import sys
import torch as _torch

from token_merging_monkey_patch.quadtree_attn_monkey_patch import (
    replace_qwen2_with_quadtree_attn,
)

# --- Fix 1: max_batch_size ---
import llava.model.builder as _builder
_orig_load = _builder.load_pretrained_model
def _patched_load(*args, **kwargs):
    result = _orig_load(*args, **kwargs)
    _tok, model, _ip, _ctx = result
    if not hasattr(model.config, 'max_batch_size'):
        model.config.max_batch_size = 32
    return result
_builder.load_pretrained_model = _patched_load

# --- Fix 2: _update_causal_mask (no-op if transformers already has it) ---
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

# --- Fix 3: image_token_start_index / image_token_length / num_frame ---
# STTM's patched Qwen2Model.forward() reads these from self (the inner LlavaQwenModel)
# at prefill time to locate and spatially reshape the image tokens. Hook into
# prepare_inputs_labels_for_multimodal which is where LLaVA inserts image embeddings
# and where the final token layout is first known.
try:
    from llava.model.llava_arch import LlavaMetaForCausalLM as _LlavaBase
    from llava.constants import IMAGE_TOKEN_INDEX as _IMG_TOK
    _orig_prepare = _LlavaBase.prepare_inputs_labels_for_multimodal

    def _sttm_prepare(self_m, input_ids, *args, **kwargs):
        result = _orig_prepare(self_m, input_ids, *args, **kwargs)
        try:
            # signature: (input_ids, position_ids, attn_mask, past_kv, labels, images, ...)
            images = args[4] if len(args) > 4 else kwargs.get('images')
            new_embeds = result[4]
            if input_ids is not None and images is not None and new_embeds is not None:
                img_mask = (input_ids == _IMG_TOK)
                if img_mask.any():
                    img_start = img_mask.nonzero()[0][1].item()
                    orig_non_img = input_ids.shape[1] - img_mask.sum().item()
                    img_tok_len = new_embeds.shape[1] - orig_non_img
                    if isinstance(images, (list, tuple)) and len(images) > 0:
                        fi = images[0]
                        if isinstance(fi, _torch.Tensor) and fi.dim() >= 4:
                            n_frames = fi.shape[0]
                        elif isinstance(fi, (list, tuple)):
                            n_frames = len(fi)
                        else:
                            n_frames = len(images)
                    elif isinstance(images, _torch.Tensor):
                        n_frames = images.shape[0]
                    else:
                        n_frames = max(1, img_tok_len // 169)
                    # Strip any non-spatial tokens (e.g. SigLIP CLS) so
                    # img_tok_len is exactly T*H*W and einops rearrange succeeds.
                    tokens_per_frame = img_tok_len // n_frames
                    img_tok_len = tokens_per_frame * n_frames
                    self_m.model.image_token_start_index = _torch.tensor(img_start, dtype=_torch.long)
                    self_m.model.image_token_length = _torch.tensor(img_tok_len, dtype=_torch.long)
                    self_m.model.num_frame = _torch.tensor(n_frames, dtype=_torch.long)
        except Exception:
            pass
        return result

    _LlavaBase.prepare_inputs_labels_for_multimodal = _sttm_prepare
    print("STTM: patched prepare_inputs_labels_for_multimodal", flush=True)
except Exception as e:
    print(f"STTM: could not patch prepare_inputs_labels_for_multimodal: {e}", flush=True)

# --- STTM quadtree patch ---
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

# --- Fix 5: prompt_stat + runtime_dict in _sample ---
# (a) STTM's generate() passes prompt_stat=None when called from lmms_eval
#     (no prompt_stat kwarg). _sample() then does
#     model_kwargs['prompt_stat']['key'] = val → TypeError on None.
# (b) STTM's _sample() always returns (sequences, runtime_dict) — a 2-tuple.
#     transformers' generate() passes this tuple back to lmms_eval which calls
#     batch_decode((tensor, dict)) and fails. Strip runtime_dict here so
#     generate() sees a normal tensor return.
try:
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM as _LlavaQwen
    _orig_sample = _LlavaQwen._sample
    def _patched_sample(self, *args, **model_kwargs):
        if model_kwargs.get('prompt_stat') is None:
            model_kwargs['prompt_stat'] = {}
        result = _orig_sample(self, *args, **model_kwargs)
        # Strip runtime_dict — STTM always returns (sequences, runtime_dict)
        if isinstance(result, tuple) and len(result) == 2:
            sequences, _runtime = result
            return sequences
        return result
    _LlavaQwen._sample = _patched_sample
    print("STTM: patched _sample (prompt_stat init + runtime_dict strip)", flush=True)
except Exception as e:
    print(f"STTM: could not patch _sample: {e}", flush=True)

# --- Fix 4: num_logits_to_keep ---
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

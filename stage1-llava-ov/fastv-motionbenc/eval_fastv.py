#!/usr/bin/env python3
"""
FastV × MotionBench — attention-based visual token pruning inside the LLM.

FastV (arXiv 2403.06764) prunes image tokens at LLM layer K based on the
average attention they receive from all other tokens. Tokens in the bottom R%
of received-attention scores are dropped; computation continues with the reduced
set through layers K+1 onward. No retraining required.

PORTING STATUS: FastV's reference implementation patches modeling_llama.py in a
custom transformers fork. LLaVA-OV uses Qwen1.5 (modeling_qwen2.py), not LLaMA.
The porting work is isolated to apply_fastv() below — everything else is ready.

Port reference: FastV/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py
Target:         Qwen2Model.forward() in transformers/models/qwen2/modeling_qwen2.py
Algorithm (identical for both):
  1. Run the first K transformer layers normally.
  2. After layer K, compute for each token the average attention it received:
       attn_score[i] = mean over heads of: sum_j attn_weight[j, i]  (column sum)
  3. Among image tokens only (indices image_start : image_start + image_len),
     keep those with attn_score in the top (1 - fastv_r) fraction.
  4. Remove the pruned tokens from the hidden state and all subsequent KV caches.
  5. Continue forward through layers K+1 ... L with the shortened sequence.

Source:  https://github.com/chenllliang/FastV
Conda:   fastv  (clone dycoke11, then install FastV's custom transformers fork)
Weights: /project/rhu/dpalfaro/weights/llava-ov-7b  (LLaVA-OV-7B, Qwen 1.5)
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm


VIDEO_BASE  = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# FastV — paper-exact port of the authors' attention-rerank pruning to Qwen2
# ---------------------------------------------------------------------------
#
# Faithful to the reference implementation:
#   FastV/src/transformers/src/transformers/models/llama/modeling_llama.py
# (the decoder-loop "Attention Rerank" block). FastV's real mechanism is NOT
# KV-cache index-dropping — it is ATTENTION MASKING:
#
#   Params (paper naming): SYS_LENGTH (prefix before image block),
#     IMAGE_TOKEN_LENGTH, ATTENTION_RANK (# image tokens to KEEP), AGG_LAYER (=K).
#   - layers < K: normal causal mask.
#   - at layer K: take the PREVIOUS layer's attention (layer_outputs[1]), average
#     over heads, take the LAST query row, slice the image span, topk(ATTENTION_RANK)
#     to pick the kept image tokens, then build a boolean keep-mask that sets every
#     OTHER image token to False. Pruned tokens stay in the sequence but every
#     layer >= K ignores them.
#   - layers > K: reuse that same generated mask.
#
# We reproduce this exactly by wrapping Qwen2Model.forward. The image span is the
# authoritative [SYS_LENGTH : SYS_LENGTH + IMAGE_TOKEN_LENGTH], where
# IMAGE_TOKEN_LENGTH is the real vision-token count LLaVA-OV computes per sample
# (self.model.lengeh_vision_token, set in prepare_inputs_labels_for_multimodal;
# see llava_arch.py). SYS_LENGTH = 14 (qwen_1_5 preamble).
#
# The paper expresses the drop as ATTENTION_RANK (keep count); our CLI takes
# fastv_r (drop fraction) for consistency with the sbatch, and converts:
#   ATTENTION_RANK = round(IMAGE_TOKEN_LENGTH * (1 - fastv_r)).

IMAGE_TOKEN_START_INDEX = 14  # SYS_LENGTH: qwen_1_5 preamble before the image block


def apply_fastv(model, fastv_k: int, fastv_r: float):
    """
    Install FastV attention-rerank pruning on this LLaVA-OV Qwen2 model by
    wrapping Qwen2Model.forward. Reproduces the authors' modeling_llama.py block.

    fastv_k: AGG_LAYER — layer index at which to rerank/prune (paper default: 2).
    fastv_r: fraction of image tokens to DROP (paper default: 0.5).

    Requires attn_implementation="eager" (needs per-layer attention weights).
    """
    import types
    import torch as _torch

    inner = model.model  # Qwen2Model
    orig_forward = inner.forward

    def fastv_forward(self, *args, **kwargs):
        # Normalize call to kwargs so we can read/inspect uniformly.
        # LlavaQwenForCausalLM calls self.model(...) with keyword args.
        inputs_embeds = kwargs.get("inputs_embeds", None)
        attention_mask = kwargs.get("attention_mask", None)
        input_ids = kwargs.get("input_ids", None)

        seq_len = None
        if inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
        elif input_ids is not None:
            seq_len = input_ids.shape[1]

        # Only rerank on the prefill pass (seq > 1). Decode steps are untouched.
        if seq_len is None or seq_len <= 1:
            return orig_forward(*args, **kwargs)

        # The vision-token count is stashed by prepare_inputs_labels_for_multimodal.
        # It may live on the inner Qwen2Model (self) OR on the outer
        # LlavaQwenForCausalLM, and DyCoke also mirrors it onto DycokeConfig.
        img_len = getattr(self, "lengeh_vision_token", None)
        if img_len is None:
            img_len = getattr(model, "lengeh_vision_token", None)
        if img_len is None:
            cfg = getattr(self, "DycokeConfig", None)
            img_len = getattr(cfg, "image_token_length", None) if cfg else None
        if isinstance(img_len, _torch.Tensor):
            img_len = int(img_len.item())
        if img_len is not None:
            img_len = int(img_len)
        if (img_len is None or img_len <= 0) and seq_len is not None:
            # Fallback: LLaVA-OV expands the single <image> placeholder into the
            # visual embeddings, so the prefill sequence is (text prompt tokens)
            # + (vision tokens). The text tail after the image block is short and
            # fixed by the template; derive the image span from the prefill length
            # minus the preamble and the tokenized question tail.
            tail = getattr(self, "_fastv_text_tail", None)
            if tail is not None and seq_len - IMAGE_TOKEN_START_INDEX - tail > 0:
                img_len = seq_len - IMAGE_TOKEN_START_INDEX - tail
        if img_len is None or img_len <= 0:
            # No vision tokens located -> cannot prune; run stock (and warn once).
            if not getattr(self, "_fastv_warned", False):
                import logging
                logging.warning("FastV: lengeh_vision_token missing; running unpruned.")
                self._fastv_warned = True
            return orig_forward(*args, **kwargs)

        sys_len = IMAGE_TOKEN_START_INDEX
        keep = max(1, int(round(img_len * (1 - fastv_r))))

        # IMPORTANT — why this uses the KV-cache route, not an attention mask:
        # this LLaVA build's patched Qwen2Model.forward calls every decoder layer
        # with attention_mask=None (DyCoke relies on cache-index pruning instead).
        # An additive-mask hook is therefore silently dropped, and forcing
        # output_attentions=True also shifts the layer_outputs tuple layout that
        # `next_decoder_cache = layer_outputs[2 if output_attentions else 1]`
        # depends on — which produced empty generations. So we express FastV's
        # policy through the mechanism this build actually honours:
        # PrunableDynamicCache.kv_cache — the list of kept token indices. Setting
        # it once makes update() gather only those tokens for all later layers and
        # every decode step, which is exactly FastV's "prune once at layer K".
        #
        # We capture layer K's attention with a hook (so we do NOT have to enable
        # output_attentions globally) and then set kv_cache.
        state = {"done": False}

        def _capture(module, args_, output):
            # Runs after layer fastv_k's self-attn. Under eager attention the
            # module returns (attn_output, attn_weights, past_kv).
            if state["done"]:
                return output
            attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attn is None:
                return output
            cache = kwargs.get("past_key_values", None)
            if cache is None or not hasattr(cache, "kv_cache"):
                return output
            # FastV ranking: average over heads, take the LAST query row, slice the
            # image span, keep the top-`keep` (== ATTENTION_RANK) image tokens.
            avg_last = attn.mean(dim=1)[0, -1]                 # (kv_len,)
            img_scores = avg_last[sys_len:sys_len + img_len]
            top = (img_scores.topk(keep).indices + sys_len).sort()[0]
            total = avg_last.shape[0]
            full = _torch.arange(total, device=attn.device)
            keep_idx = _torch.cat([full[:sys_len], top, full[sys_len + img_len:]])
            cache.kv_cache = keep_idx.tolist()
            state["done"] = True
            if not getattr(self, "_fastv_logged", False):
                import logging
                logging.warning(
                    "FastV ACTIVE: img_len=%d keep=%d (dropped %d) seq=%d",
                    img_len, keep, img_len - keep, total)
                self._fastv_logged = True
            self._fastv_pruned_tokens = img_len - keep
            return output

        # Only layer K needs real attention weights. Loading the WHOLE model as
        # eager breaks generation on this build (A/B job 7769959 -> empty output),
        # so we temporarily route just this one module through the eager forward
        # and restore it immediately afterwards.
        target = self.layers[fastv_k].self_attn
        h_cap = target.register_forward_hook(_capture, with_kwargs=False)

        def _want_attn(module, args_, kwargs_):
            kw = dict(kwargs_)
            kw["output_attentions"] = True
            return (args_, kw)

        h_pre = target.register_forward_pre_hook(_want_attn, with_kwargs=True)

        # Swap this module's class to the eager implementation for the call.
        prev_cls = type(target)
        eager_cls = None
        try:
            from transformers.models.qwen2 import modeling_qwen2 as _mq
            eager_cls = getattr(_mq, "Qwen2Attention", None)
        except Exception:
            eager_cls = None
        swapped = False
        if eager_cls is not None and prev_cls is not eager_cls:
            try:
                target.__class__ = eager_cls
                swapped = True
            except Exception:
                swapped = False
        try:
            return orig_forward(*args, **kwargs)
        finally:
            if swapped:
                target.__class__ = prev_cls
            h_cap.remove()
            h_pre.remove()

    inner.forward = types.MethodType(fastv_forward, inner)
    model.config._fastv_enabled = True
    model.config._fastv_k = fastv_k
    model.config._fastv_r = fastv_r
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, fastv_k: int, fastv_r: float, enable_fastv: bool):
    from llava.model.builder import load_pretrained_model

    # NOTE: do NOT load the whole model with attn_implementation="eager".
    # Verified by A/B test (job 7769959): loading this LLaVA-OV build with eager
    # attention produces EMPTY generations for every sample, while sdpa produces
    # correct output — independently of FastV. FastV instead flips only layer K's
    # attention module to eager at runtime (see apply_fastv), so the rest of the
    # model keeps the working sdpa path.
    attn_impl = "sdpa"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        attn_implementation=attn_impl,
    )
    if enable_fastv:
        model = apply_fastv(model, fastv_k=fastv_k, fastv_r=fastv_r)
    model = model.cuda()
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Video loading — subprocess-isolated for NFS stale-handle safety
# ---------------------------------------------------------------------------
def load_frames(video_path: str, num_frames: int):
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0))
            total = len(vr)
            indices = np.linspace(0, total - 1, n, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            q.put(("ok", frames))
        except Exception as e:
            q.put(("error", str(e)))

    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frames, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill()
        proc.join(timeout=5)
        raise RuntimeError(f"load_frames: timeout (NFS stale?): {video_path}")
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == "error":
        raise RuntimeError(f"load_frames: decode error: {video_path} — {data}")

    from PIL import Image
    return [Image.fromarray(f) for f in data]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames, question,
                  conv_template: str = "qwen_2"):
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # Tell FastV how many TEXT tokens follow the image block, so it can locate the
    # visual span even when lengeh_vision_token isn't populated. input_ids holds
    # the prompt with a single IMAGE_TOKEN_INDEX placeholder; everything after it
    # is the question + assistant tag.
    try:
        ids = input_ids[0].tolist()
        img_pos = ids.index(IMAGE_TOKEN_INDEX)
        model.model._fastv_text_tail = len(ids) - img_pos - 1
    except (ValueError, AttributeError):
        pass

    images = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    images = images.to(dtype=model.dtype, device="cuda")

    w, h = frames[0].size
    image_sizes = [(h, w)] * len(frames)

    output_ids = model.generate(
        input_ids,
        images=[images],
        image_sizes=image_sizes,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=16,
        use_cache=True,
    )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def find_video(video_path: str):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


def score_prediction(prediction: str, ground_truth: str):
    gt = ground_truth.strip().upper()
    if gt == "NA":
        return None
    match = re.search(r"\b([A-D])\b", prediction.upper())
    pred = match.group(1) if match else prediction.strip().upper()[:1]
    return int(pred == gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--conv_template", default="qwen_2",
                        help="must match the backbone (DyCoke uses qwen_2)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fastv", action="store_true",
                        help="Apply FastV pruning (requires Qwen port — see apply_fastv())")
    parser.add_argument("--fastv_k", type=int, default=2,
                        help="LLM layer at which to prune (FastV paper default: 2)")
    parser.add_argument("--fastv_r", type=float, default=0.5,
                        help="Fraction of image tokens to drop (FastV paper default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(
        args.model_path,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        enable_fastv=args.fastv,
    )

    samples = []
    with open(args.meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]
    print(f"Evaluating {len(samples)} samples", flush=True)

    results = []
    scores  = []

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        video_path   = find_video(sample["video_path"])
        question     = sample["qa"][0]["question"]
        ground_truth = sample["qa"][0]["answer"]
        q_type       = sample.get("question_type", "Unknown")

        if video_path is None:
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            prediction = ""
        else:
            try:
                frames     = load_frames(video_path, args.num_frames)
                prediction = run_inference(tokenizer, model, image_processor, frames, question,
                                           conv_template=args.conv_template)
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}", file=sys.stderr)
                prediction = ""
                torch.cuda.empty_cache()

        s = score_prediction(prediction, ground_truth)
        results.append({
            "idx":           i,
            "video_path":    sample["video_path"],
            "question_type": q_type,
            "ground_truth":  ground_truth,
            "prediction":    prediction,
            "correct":       s,
        })
        if s is not None:
            scores.append(s)

    out_file = os.path.join(args.output_dir, "results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total    = len(scores)
    correct  = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "accuracy":         accuracy,
        "correct":          correct,
        "total_scoreable":  total,
        "total_na_skipped": na_count,
        "total_samples":    len(results),
        "model":            args.model_path,
        "fastv_params": {
            "enabled":  args.fastv,
            "fastv_k":  args.fastv_k,
            "fastv_r":  args.fastv_r,
            "num_frames": args.num_frames,
        },
    }
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)", flush=True)

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:  {out_file}", flush=True)
    print(f"Summary:  {summary_file}", flush=True)


if __name__ == "__main__":
    main()

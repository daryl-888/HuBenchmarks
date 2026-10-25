#!/usr/bin/env python3
"""
FastV x MotionBench — attention-based visual token pruning inside the LLM.
FastV (arXiv 2403.06764) prunes image tokens at LLM layer K based on the
average attention they receive from all other tokens. Tokens in the bottom R%
of received-attention scores are dropped; computation continues with the reduced
set through layers K+1 onward. No retraining required.

CACHE-FIX VARIANT: identical to eval_fastv.py except for patch_prunable_cache()
below. Root-cause bug (see docs/notes/llava_ov_kv_cache_bug.md): DyCoke's
PrunableDynamicCache.kv_cache is a FIXED index snapshot set once when pruning
fires. update() keeps torch.cat-ing newly generated tokens onto the underlying
key/value tensors every decode step, but the gather that actually gets RETURNED
to attention only ever uses that original frozen index list — so every token
generated after the prune point is permanently invisible to self-attention at
layers >= the prune layer, for the rest of decoding. Because FastV prunes at
PREFILL (before any output token exists), this blinds the model to its own
output for the entire 16-token generation on every sample -- independent of how
much is kept, which is why the retention sweep (10-75%) is flat at ~36%.

Fix: extend kv_cache with each newly-appended token's position on every decode
step, so the model regains the ability to see tokens it already generated,
while the ORIGINALLY PRUNED image tokens stay dropped forever (that part of the
mechanism is correct and unchanged).

PORTING STATUS: FastV's reference implementation patches modeling_llama.py in a
custom transformers fork. LLaVA-OV uses Qwen1.5 (modeling_qwen2.py), not LLaMA.
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

VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# THE FIX — patch PrunableDynamicCache.update to stop freezing kv_cache
# ---------------------------------------------------------------------------
def patch_prunable_cache():
    import sys as _sys
    import torch as _torch
    try:
        from llava.model.language_model import modeling_qwen2 as _mq
    except Exception as e:
        print(f"CAPFIX IMPORT FAILED: {type(e).__name__}: {e}", file=_sys.stderr, flush=True)
        raise
    Cache = _mq.PrunableDynamicCache
    print(f"CAPFIX: found PrunableDynamicCache at {_mq.__file__} id={id(Cache)}",
          file=_sys.stderr, flush=True)
    if getattr(Cache, "_capfix_patched", False):
        print("CAPFIX: already patched, skipping", file=_sys.stderr, flush=True)
        return

    # SECOND BUG, found after the physical-prune fix above still crashed ~23%
    # of samples with shape mismatches: Qwen2Model.forward() ends with
    #   next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    # use_legacy_cache is True whenever past_key_values arrived as None (i.e.
    # every FIRST call of a generate()), so the cache gets serialized to a
    # plain tuple and, on the very next step, from_legacy_cache() rebuilds a
    # BRAND NEW cache object from it. That reconstruction resets _seen_tokens
    # to the tuple's tensor length (now physically pruned), so the cache
    # believes it's only ~1000 tokens into a sequence that's actually ~6000+
    # tokens along -- every downstream computation keyed on "how far are we"
    # (slicing which input_ids are "new", rotary position bookkeeping) goes
    # inconsistent, which is what produced the shape-mismatch crashes.
    # Fix: never let the cache serialize to a tuple in the first place. Then
    # the SAME cache object persists for the whole generate() call, kv_cache/
    # _capfix_pruned_layers/_seen_tokens all stay self-consistent throughout,
    # and from_legacy_cache() is never invoked at all.
    orig_to_legacy = Cache.to_legacy_cache
    def to_legacy_cache_noop(self):
        return self
    Cache.to_legacy_cache = to_legacy_cache_noop
    print("CAPFIX: PrunableDynamicCache.to_legacy_cache PATCHED to no-op "
          "(cache identity persists across generate() steps, no reconstruction)",
          file=_sys.stderr, flush=True)

    # ROOT CAUSE (confirmed via live instrumentation, see paper notes): the
    # original code's gather-on-return only prunes what's handed to attention
    # for the CURRENT call; the stored self.key_cache/self.value_cache always
    # keep the FULL, un-pruned tensors (torch.cat with the raw key_states).
    # When a fresh generate() call's past_key_values is None, Qwen2Model.
    # forward() ends by exporting the cache via to_legacy_cache(), which reads
    # that FULL, un-pruned storage -- the prune is silently lost at this
    # export. The next generation step reconstructs a brand-new
    # PrunableDynamicCache from that legacy (unpruned) tuple via
    # from_legacy_cache(), and FastV's hook fails to re-engage on this second
    # pass (attn comes back None), so ALL real decoding then proceeds on the
    # entirely unpruned sequence. Net effect: FastV's token pruning had ZERO
    # effect on the actually-generated text in every run to date.
    #
    # Fix: make the prune PHYSICAL, not logical. The first time a layer's
    # update() is called after kv_cache is set, gather its STORED tensors
    # in-place (not just the returned view) and mark that layer done. A
    # physically-shrunk cache survives to_legacy_cache()/from_legacy_cache()
    # round-trips intact, since there's no separate index list to lose --
    # the tensors themselves are already the pruned size.
    def fixed_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if id(self) not in _stats["seen_ids"] and layer_idx == 0:
            _stats["seen_ids"].add(id(self))
            print(f"CAPFIX: new cache instance id={id(self)} (#{len(_stats['seen_ids'])} seen so far), "
                  f"key_shape={tuple(key_states.shape)}", file=_sys.stderr, flush=True)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = _torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = _torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if not hasattr(self, "_capfix_pruned_layers"):
            self._capfix_pruned_layers = set()
        if self.kv_cache is not None and layer_idx not in self._capfix_pruned_layers:
            n = self.key_cache[layer_idx].shape[-2]
            keep = [i for i in self.kv_cache if i < n]
            idx = _torch.tensor(keep, device=self.key_cache[layer_idx].device)
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(2, idx)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(2, idx)
            self._capfix_pruned_layers.add(layer_idx)
            if not _stats["first_seen"]:
                print(f"CAPFIX: physically pruned layer={layer_idx} from {n} to {len(keep)} tokens "
                      f"(permanent, survives cache export)", file=_sys.stderr, flush=True)
                _stats["first_seen"] = True
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    _stats = {"first_seen": False, "seen_ids": set()}
    Cache.update = fixed_update
    Cache._capfix_patched = True
    print("CAPFIX: PrunableDynamicCache.update PATCHED successfully (physical-prune variant)",
          file=_sys.stderr, flush=True)

    orig_from_legacy = Cache.from_legacy_cache.__func__
    def from_legacy_cache_traced(cls, past_key_values=None):
        print(f"CAPFIX: from_legacy_cache CALLED (should only happen once, at the very "
              f"first generate() step) past_key_values={'None' if past_key_values is None else type(past_key_values).__name__}",
              file=_sys.stderr, flush=True)
        return orig_from_legacy(cls, past_key_values)
    Cache.from_legacy_cache = classmethod(from_legacy_cache_traced)


# ---------------------------------------------------------------------------
# FastV — paper-exact port of the authors' attention-rerank pruning to Qwen2
# (unchanged from eval_fastv.py — the bug and the fix live entirely in the
# cache class, not in FastV's own hook logic)
# ---------------------------------------------------------------------------
IMAGE_TOKEN_START_INDEX = 14  # SYS_LENGTH: qwen_1_5 preamble before the image block

def apply_fastv(model, fastv_k: int, fastv_r: float):
    import types
    import torch as _torch
    inner = model.model  # Qwen2Model
    orig_forward = inner.forward

    def fastv_forward(self, *args, **kwargs):
        if not getattr(self, "_fastv_entered", False):
            import logging
            def _d(x):
                return (f"T{tuple(x.shape)}{'f' if x.is_floating_point() else 'i'}"
                        if isinstance(x, _torch.Tensor) else type(x).__name__)
            logging.warning("FastV: ENTERED args=[%s] kwargs={%s}",
                            ", ".join(_d(a) for a in args),
                            ", ".join(f"{k}:{_d(v)}" for k, v in kwargs.items()))
            self._fastv_entered = True
        inputs_embeds = kwargs.get("inputs_embeds", None)
        input_ids = kwargs.get("input_ids", None)
        if inputs_embeds is None and input_ids is None:
            for a in args:
                if not isinstance(a, _torch.Tensor):
                    continue
                if a.dim() == 3 and a.is_floating_point() and inputs_embeds is None:
                    inputs_embeds = a
                elif a.dim() == 2 and not a.is_floating_point() and input_ids is None:
                    input_ids = a
        seq_len = None
        if inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
        elif input_ids is not None:
            seq_len = input_ids.shape[1]
        if seq_len is None or seq_len <= 1:
            return orig_forward(*args, **kwargs)
        img_len = kwargs.get("lengeh_vision_token", None)
        if img_len is None:
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
            tail = getattr(self, "_fastv_text_tail", None)
            if tail is not None and seq_len - IMAGE_TOKEN_START_INDEX - tail > 0:
                img_len = seq_len - IMAGE_TOKEN_START_INDEX - tail
        if img_len is None or img_len <= 0:
            if not getattr(self, "_fastv_warned", False):
                import logging
                cfg = getattr(self, "DycokeConfig", None)
                logging.warning(
                    "FastV UNPRUNED: seq_len=%s inner.lengeh=%r outer.lengeh=%r "
                    "DycokeConfig=%r cfg.image_token_length=%r text_tail=%r "
                    "cache=%s",
                    seq_len,
                    getattr(self, "lengeh_vision_token", "ABSENT"),
                    getattr(model, "lengeh_vision_token", "ABSENT"),
                    "present" if cfg else "ABSENT",
                    getattr(cfg, "image_token_length", "ABSENT") if cfg else "n/a",
                    getattr(self, "_fastv_text_tail", "ABSENT"),
                    type(kwargs.get("past_key_values", None)).__name__,
                )
                self._fastv_warned = True
            return orig_forward(*args, **kwargs)
        sys_len = IMAGE_TOKEN_START_INDEX
        keep = max(1, int(round(img_len * (1 - fastv_r))))
        state = {"done": False}
        def _capture(module, args_, kwargs_, output):
            if state["done"]:
                return output
            attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attn is None:
                return output
            cache = None
            for cand in (kwargs.get("past_key_values", None),
                         getattr(module, "past_key_value", None),
                         kwargs_.get("past_key_value", None) if isinstance(kwargs_, dict) else None):
                if cand is not None and hasattr(cand, "kv_cache"):
                    cache = cand
                    break
            if cache is None:
                for a in list(args) + list(args_ or ()):
                    if hasattr(a, "kv_cache"):
                        cache = a
                        break
            if cache is None:
                if not getattr(self, "_fastv_nocache", False):
                    import sys as _sys2
                    print(f"CAPFIX-DBG: FastV no cache reachable at layer {fastv_k} "
                          f"(module={type(module).__name__})", file=_sys2.stderr, flush=True)
                    self._fastv_nocache = True
                return output
            avg_last = attn.mean(dim=1)[0, -1]                 # (kv_len,)
            img_scores = avg_last[sys_len:sys_len + img_len]
            top = (img_scores.topk(keep).indices + sys_len).sort()[0]
            total = avg_last.shape[0]
            full = _torch.arange(total, device=attn.device)
            keep_idx = _torch.cat([full[:sys_len], top, full[sys_len + img_len:]])
            cache.kv_cache = keep_idx.tolist()
            # Layers BEFORE this one already had their one-and-only update() call
            # for this prefill (with kv_cache still None, so they stored the FULL
            # sequence) -- they won't be called again until the next generation
            # step, by which point fixed_update's per-layer gate would think
            # they're done-for-this-pass too late. Prune them retroactively, RIGHT
            # NOW, so every layer's stored cache ends this forward pass at the
            # SAME (pruned) length -- a length-mismatched cache across layers is
            # almost certainly what forces transformers to discard/reconstruct it.
            if not hasattr(cache, "_capfix_pruned_layers"):
                cache._capfix_pruned_layers = set()
            keep_list = keep_idx.tolist()
            for li in range(len(cache.key_cache)):
                if li in cache._capfix_pruned_layers:
                    continue
                n = cache.key_cache[li].shape[-2]
                keep_here = [i for i in keep_list if i < n]
                idx_t = _torch.tensor(keep_here, device=cache.key_cache[li].device)
                cache.key_cache[li] = cache.key_cache[li].index_select(2, idx_t)
                cache.value_cache[li] = cache.value_cache[li].index_select(2, idx_t)
                cache._capfix_pruned_layers.add(li)
            print(f"CAPFIX-DBG: retroactively pruned {len(cache._capfix_pruned_layers)} "
                  f"already-populated layers to match", file=__import__("sys").stderr, flush=True)
            state["done"] = True
            if not getattr(self, "_fastv_logged", False):
                import sys as _sys2
                print(f"CAPFIX-DBG: FastV ACTIVE id(cache)={id(cache)} cache_type={type(cache).__name__} "
                      f"img_len={img_len} keep={keep} dropped={img_len - keep} seq={total} "
                      f"kv_cache_set_len={len(cache.kv_cache)}", file=_sys2.stderr, flush=True)
                self._fastv_logged = True
            self._fastv_pruned_tokens = img_len - keep
            return output
        target = self.layers[fastv_k].self_attn
        h_cap = target.register_forward_hook(_capture, with_kwargs=True)
        def _want_attn(module, args_, kwargs_):
            kw = dict(kwargs_)
            kw["output_attentions"] = True
            return (args_, kw)
        h_pre = target.register_forward_pre_hook(_want_attn, with_kwargs=True)
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
    import logging as _lg
    _lg.warning("FastV: installing wrapper on %s (k=%d, r=%.2f)",
                type(inner).__name__, fastv_k, fastv_r)
    inner.forward = types.MethodType(fastv_forward, inner)
    model.config._fastv_enabled = True
    model.config._fastv_k = fastv_k
    model.config._fastv_r = fastv_r
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, fastv_k: int, fastv_r: float, enable_fastv: bool):
    patch_prunable_cache()
    from llava.model.builder import load_pretrained_model
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
    parser.add_argument("--conv_template", default="qwen_2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fastv", action="store_true")
    parser.add_argument("--fastv_k", type=int, default=2)
    parser.add_argument("--fastv_r", type=float, default=0.5)
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
            "cache_bug_fix": True,
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

#!/usr/bin/env python3
"""
Qwen3-VL Baseline × MotionBench — ovqwen3.

Qwen3VLForConditionalGeneration is a native HuggingFace model, NOT a LLaVA fork.
Uses transformers AutoProcessor + from_pretrained (no llava.model.builder).

Backbone: Qwen/Qwen3-VL-8B-Instruct
Weights: /project/rhu/dpalfaro/weights/qwen3-vl-8b
Uses: AutoProcessor for video preprocessing, model.generate() for inference

This is a baseline eval — no model compression applied. All ovqwen3 models
start here, then compression methods are ported later.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Model loading — Qwen3-VL native HuggingFace
# ---------------------------------------------------------------------------

def _attn_last_row(module, inp, kv_len, kwargs=None):
    """
    Recover the LAST query row of attention weights from a Qwen3-VL attention
    module, without eager attention.

    Under sdpa `attn_weights` is None, and loading the whole model eager breaks
    generation (verified: 100% empty output). But FastV/DyCoke/HoliTom are DEFINED
    by attention ranking, so substituting a different signal would not be the
    method. Instead we recompute just the row we need:

        softmax( q_last · Kᵀ / sqrt(d) )   averaged over heads

    That is one (1 × kv_len) row — negligible cost — and is exactly the quantity
    the papers rank by.
    """
    import torch as _t
    import logging
    # Qwen3-VL calls the attention module with hidden_states as a KWARG, so the
    # positional args tuple is EMPTY (verified: inp_type=tuple len=0). Look in
    # kwargs first, then fall back to positional.
    hs = None
    if kwargs:
        hs = kwargs.get("hidden_states")
    if hs is None and isinstance(inp, (tuple, list)) and len(inp):
        hs = inp[0]
    if not getattr(module, "_attndiag", False):
        logging.warning("ATTNDIAG: inp_type=%s len=%s hs=%s dim=%s has_qproj=%s head_dim=%s",
                        type(inp).__name__, len(inp) if hasattr(inp,"__len__") else None,
                        type(hs).__name__,
                        getattr(hs,"dim",lambda:None)() if hs is not None else None,
                        hasattr(module,"q_proj"), getattr(module,"head_dim",None))
        module._attndiag = True
    if hs is None or not hasattr(hs, "dim") or hs.dim() != 3:
        return None
    try:
        q = module.q_proj(hs)                      # (b, s, n_q*d)
        k = module.k_proj(hs)                      # (b, s, n_kv*d)
    except Exception:
        return None
    b, s, _ = q.shape
    d = getattr(module, "head_dim", None)
    if not d:
        return None
    nq, nkv = q.shape[-1] // d, k.shape[-1] // d
    q = q.view(b, s, nq, d).transpose(1, 2)        # (b, nq, s, d)
    k = k.view(b, s, nkv, d).transpose(1, 2)       # (b, nkv, s, d)
    if nq != nkv:                                   # GQA: repeat kv heads
        k = k.repeat_interleave(nq // nkv, dim=1)
    ql = q[:, :, -1:, :].float()                   # last query only
    scores = _t.matmul(ql, k.float().transpose(-1, -2)) / (d ** 0.5)
    w = _t.softmax(scores, dim=-1)                 # (b, nq, 1, s)
    return w.mean(dim=1)[0, -1][:kv_len]           # (kv_len,) head-averaged


def apply_fastv(model, fastv_k: int = 2, fastv_r: float = 0.85):
    """
    FastV (arXiv 2403.06764) ported to Qwen3-VL.

    Paper mechanism, from the authors' modeling_llama.py "Attention Rerank" block:
      - run layers < K normally
      - at layer K, average the previous layer's attention over heads, take the
        LAST query row, restrict to VISUAL token positions, keep the top
        ATTENTION_RANK of them, and mask out the rest for every layer >= K.

    Qwen3-VL differs from LLaVA-OV in one helpful way: the visual token positions
    are given explicitly by `visual_pos_masks` (bool [B, S]) passed into
    Qwen3VLTextModel.forward — so unlike the LLaVA-OV port we do NOT have to infer
    the image span from a fixed prefix length. Everything else is the same policy.

    fastv_k: AGG_LAYER (paper default 2)
    fastv_r: fraction of visual tokens to DROP; 0.85 keeps 15% (project standard)
    """
    import types
    import torch as _torch

    lm = model.model.language_model  # Qwen3VLTextModel
    orig_forward = lm.forward

    def fastv_forward(self, *args, **kwargs):
        if not getattr(self, "_fastv_entered", False):
            import logging
            logging.warning("FastV(Qwen3-VL): wrapper ENTERED, kwargs=%s args=%d",
                            list(kwargs.keys()), len(args))
            self._fastv_entered = True
        vis_mask = kwargs.get("visual_pos_masks", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        # Qwen3VLTextModel may receive these positionally too.
        if inputs_embeds is None:
            for a in args:
                if hasattr(a, "dim") and a.dim() == 3 and a.is_floating_point():
                    inputs_embeds = a
                    break
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else None

        # Prefill only (multi-token). Decode steps are untouched.
        if seq_len is None or seq_len <= 1 or vis_mask is None:
            return orig_forward(*args, **kwargs)

        vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
        n_vis = vis_idx.numel()
        if n_vis == 0:
            return orig_forward(*args, **kwargs)
        keep = max(1, int(round(n_vis * (1 - fastv_r))))

        state = {"done": False, "add": None}

        def _rank(module, inp, kwargs_, out):
            # Capture layer K-1's attention weights to rank visual tokens.
            if state["done"]:
                return out
            # Qwen3VLTextAttention.forward returns (attn_output, attn_weights).
            # attn_weights is None unless eager attention is active — if that
            # happens FastV would silently no-op, so warn loudly instead.
            attn = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            # sdpa returns attn_weights=None. Recompute just the last query
            # row from q/k — the exact quantity the paper ranks by.
            recv = (attn.mean(dim=1)[0, -1] if attn is not None
                    else _attn_last_row(module, inp, seq_len, kwargs_))
            if recv is None:
                if not getattr(self, "_fastv_noattn", False):
                    import logging
                    logging.warning(
                        "FastV(Qwen3-VL): layer %d returned no attention weights "
                        "under sdpa+output_attentions — NOT PRUNING. If this fires, "
                        "the port needs a per-module eager swap.", fastv_k - 1)
                    self._fastv_noattn = True
                return out
            scores = recv[vis_idx]
            top = vis_idx[scores.topk(keep).indices]
            drop = _torch.ones(seq_len, dtype=_torch.bool, device=recv.device)
            drop[:] = False
            drop[vis_idx] = True
            drop[top] = False                               # keep the winners
            add = _torch.zeros((1, 1, 1, seq_len), dtype=_torch.float32,
                               device=recv.device)
            add[0, 0, 0, drop] = _torch.finfo(_torch.float32).min
            state["add"] = add
            state["done"] = True
            if not getattr(self, "_fastv_logged", False):
                import logging
                logging.warning("FastV(Qwen3-VL) ACTIVE: visual=%d keep=%d "
                                "(dropped %d) seq=%d layer_k=%d",
                                n_vis, keep, n_vis - keep, seq_len, fastv_k)
                self._fastv_logged = True
            return out

        def _mask(module, a_, k_):
            # For layers >= K: add -inf on the pruned visual columns.
            if state["add"] is None:
                return None
            kw = dict(k_)
            am = kw.get("attention_mask", None)
            add = state.get("add")
            if add is None:
                return None
            # BUILD-MASK: under sdpa Qwen3-VL passes attention_mask=None (pure
            # causal), positionally. Adding to None silently dropped our pruning
            # mask (port printed ACTIVE but output was byte-identical).
            # Two things must be right or generation breaks:
            #   * this hook also fires on DECODE steps where q_len == 1 (not
            #     seq_len) -- a fixed (S,S) mask is wrong there.
            #   * the model runs bf16; float32's finfo.min overflows bf16 to -inf
            #     and yields NaNs, so build the mask in the hidden-state dtype.
            # hidden_states may arrive positionally OR as a kwarg (Qwen3-VL uses
            # kwargs — verified ATTNDIAG len=0), so check both.
            hs = kw.get("hidden_states")
            if hs is None and len(a_) > 0 and hasattr(a_[0], "dim"):
                hs = a_[0]
            if hs is None or hs.dim() != 3:
                return None
            q_len = hs.shape[1]
            kv_len = seq_len
            dt = hs.dtype
            neg = _torch.finfo(dt).min
            add_row = add.to(dt)[..., :kv_len]                 # (1,1,1,kv)
            if am is None:
                if q_len == 1:
                    # decode: attend to all cached kv, minus the pruned columns
                    m = _torch.zeros((1, 1, 1, kv_len), dtype=dt, device=hs.device)
                else:
                    m = _torch.full((q_len, kv_len), neg, dtype=dt, device=hs.device)
                    m = _torch.triu(m, diagonal=kv_len - q_len + 1).unsqueeze(0).unsqueeze(0)
                am = m
            elif am.dim() != 4:
                return None
            kw["attention_mask"] = am + add_row
            if len(a_) >= 2 and hasattr(a_[1], "dim"):
                a_ = list(a_); a_[1] = kw["attention_mask"]; a_ = tuple(a_)
            return (a_, kw)

        handles = []
        if fastv_k - 1 >= 0:
            handles.append(self.layers[fastv_k - 1].self_attn
                           .register_forward_hook(_rank, with_kwargs=True))
        for i in range(fastv_k, len(self.layers)):
            handles.append(self.layers[i]
                           .register_forward_pre_hook(_mask, with_kwargs=True))
        # need attention weights at layer K-1
        prev = kwargs.get("output_attentions", None)
        kwargs["output_attentions"] = True
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
            if prev is None:
                kwargs.pop("output_attentions", None)

    lm.forward = types.MethodType(fastv_forward, lm)
    return model


def load_model(model_path: str, enable_fastv: bool = False,
               fastv_k: int = 2, fastv_r: float = 0.85):
    """Load Qwen3-VL via transformers from_pretrained (no LLaVA)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # eager broke generation (100% empty output) on the smoke — load sdpa and
        # request attentions per-call instead (Qwen3 attention honours
        # output_attentions=True even under sdpa by falling back for that call).
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if enable_fastv:
        model = apply_fastv(model, fastv_k=fastv_k, fastv_r=fastv_r)
    model.eval()
    return None, model, processor  # Qwen3 uses processor, not tokenizer + image_processor separately


# ---------------------------------------------------------------------------
# Video loading — standard subprocess-isolated
# ---------------------------------------------------------------------------
def load_frames(video_path: str, num_frames: int) -> list:
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
# Inference — Qwen3-VL native generate()
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(model, processor, frames: list, question: str,
                  num_frames: int = 32) -> str:
    """
    Qwen3-VL inference using native chat template + video preprocessing.
    Handles video via processor with temporal patch support.
    """
    # Build conversation using Qwen3's chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": question + POST_PROMPT},
            ],
        }
    ]

    # Apply chat template to build prompt
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process video + text together
    # CRITICAL: Qwen3VLVideoProcessor has do_sample_frames=True and fps=2, so by
    # default it RE-SAMPLES whatever frame list we hand it, ignoring --num_frames.
    # With no video_metadata it also warns "Defaulting to fps=24". We already
    # sampled exactly num_frames uniformly in load_video, so turn the processor's
    # own sampling OFF and let it consume our frames verbatim.
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames],
        return_tensors="pt",
        do_sample_frames=False,
    )
    # Report the frame count that ACTUALLY reaches the model, once, so a silent
    # re-sample can never go unnoticed again.
    if not getattr(run_inference, "_frames_logged", False):
        import logging
        pvg = inputs.get("pixel_values_videos", None)
        gt = inputs.get("video_grid_thw", None)
        logging.warning("Qwen3-VL FRAMES: requested=%d given=%d "
                        "pixel_values_videos=%s video_grid_thw=%s",
                        num_frames, len(frames),
                        tuple(pvg.shape) if pvg is not None else None,
                        gt.tolist() if gt is not None else None)
        run_inference._frames_logged = True

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=16,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


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
    parser.add_argument("--fastv", action="store_true", help="apply FastV pruning")
    parser.add_argument("--fastv_k", type=int, default=2, help="AGG_LAYER (paper: 2)")
    parser.add_argument("--fastv_r", type=float, default=0.85,
                        help="fraction of visual tokens to DROP; 0.85 keeps 15%")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3-VL model...", flush=True)
    _, model, processor = load_model(args.model_path, enable_fastv=args.fastv,
                                     fastv_k=args.fastv_k, fastv_r=args.fastv_r)

    samples = []
    with open(args.meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    print(f"Evaluating {len(samples)} samples", flush=True)

    results = []
    scores = []
    per_category = {}

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        video_path = find_video(sample["video_path"])
        question = sample["qa"][0]["question"]
        ground_truth = sample["qa"][0]["answer"]
        q_type = sample.get("question_type", "Unknown")

        if video_path is None:
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            prediction = ""
        else:
            try:
                frames = load_frames(video_path, args.num_frames)
                prediction = run_inference(
                    model, processor, frames, question,
                    num_frames=args.num_frames,
                )
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}",
                      file=sys.stderr)
                prediction = ""
                torch.cuda.empty_cache()

        s = score_prediction(prediction, ground_truth)
        results.append({
            "idx": i,
            "video_path": sample["video_path"],
            "question_type": q_type,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": s,
        })
        if s is not None:
            scores.append(s)
            per_category[q_type] = per_category.get(q_type, {"correct": 0, "total": 0})
            per_category[q_type]["total"] += 1
            per_category[q_type]["correct"] += s

    out_file = os.path.join(args.output_dir, "results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(scores)
    correct = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total_scoreable": total,
        "total_na_skipped": na_count,
        "total_samples": len(results),
        "fastv_params": {"enabled": bool(args.fastv), "fastv_k": args.fastv_k,
                         "fastv_r": args.fastv_r,
                         "keeps": f"{(1-args.fastv_r)*100:.0f}% of visual tokens"},
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "num_frames": args.num_frames,
        "note": "FastV on Qwen3-VL — attention-rerank pruning at layer fastv_k",
        "per_category": per_category,
    }
    print(
        f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)",
        flush=True,
    )
    for cat in sorted(per_category.keys()):
        c = per_category[cat]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0.0
        print(f"  {cat}: {c['correct']}/{c['total']} = {acc:.4f}", flush=True)

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:  {out_file}", flush=True)
    print(f"Summary:  {summary_file}", flush=True)


if __name__ == "__main__":
    main()

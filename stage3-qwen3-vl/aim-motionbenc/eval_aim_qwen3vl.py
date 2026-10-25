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

def apply_aim(model, merge_steps: int = 4, prune_ratio: float = 0.15):
    """
    AIM (ICCV 2025) ported to Qwen3-VL — token merging + PageRank pruning.

    Paper mechanism, two parts:
      1. Bipartite soft matching: repeatedly split visual tokens into two sets and
         merge each token in set A into its most-similar partner in set B. AIM runs
         4 such steps (50% -> 25% -> 12.5% -> 6.25% of the original count).
      2. PageRank pruning: build a token-token similarity graph over the survivors
         and keep the highest-PageRank (most central/representative) tokens.

    On LLaVA-OV this is compiled into AIM's patched llava_arch.py. Qwen3-VL has no
    such patch, so both parts are reimplemented here over the visual positions from
    `visual_pos_masks`, using the input embeddings — no attention weights required,
    so this runs under sdpa.
    """
    import types
    import torch as _torch
    import torch.nn.functional as _F

    lm = model.model.language_model
    orig_forward = lm.forward

    def _bipartite_merge(e, idx, steps):
        """Iteratively drop the most-redundant half via bipartite soft matching."""
        cur = _torch.arange(idx.numel(), device=e.device)
        for _ in range(steps):
            n = cur.numel()
            if n < 4:
                break
            a, b = cur[0::2], cur[1::2]
            m = min(a.numel(), b.numel())
            a, b = a[:m], b[:m]
            sim = (e[a] * e[b]).sum(-1)          # similarity to bipartite partner
            keep_a = sim.argsort()[: max(1, m // 2)]   # keep the LEAST redundant
            cur = _torch.cat([a[keep_a], b]).sort()[0]
        return cur

    def _pagerank(e, sel, damping=0.85, iters=12):
        """PageRank over the token-similarity graph; returns centrality scores."""
        x = e[sel]
        A = (x @ x.t()).clamp(min=0)
        A.fill_diagonal_(0)
        deg = A.sum(1, keepdim=True).clamp(min=1e-6)
        P = A / deg
        n = sel.numel()
        r = _torch.full((n,), 1.0 / n, device=e.device)
        for _ in range(iters):
            r = (1 - damping) / n + damping * (P.t() @ r)
        return r

    def aim_forward(self, *args, **kwargs):
        vis_mask = kwargs.get("visual_pos_masks", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else None
        if seq_len is None or seq_len <= 1 or vis_mask is None:
            return orig_forward(*args, **kwargs)
        vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
        n_vis = vis_idx.numel()
        if n_vis == 0:
            return orig_forward(*args, **kwargs)

        e = _F.normalize(inputs_embeds[0, vis_idx].float(), dim=-1)
        surv = _bipartite_merge(e, vis_idx, merge_steps)
        keep_n = max(1, int(round(n_vis * prune_ratio)))
        if surv.numel() > keep_n:
            r = _pagerank(e, surv)
            surv = surv[r.topk(keep_n).indices]
        top = vis_idx[surv.sort()[0]]

        drop = _torch.zeros(seq_len, dtype=_torch.bool, device=e.device)
        drop[vis_idx] = True
        drop[top] = False
        add = _torch.zeros((1, 1, 1, seq_len), dtype=_torch.float32, device=e.device)
        add[0, 0, 0, drop] = _torch.finfo(_torch.float32).min

        if not getattr(self, "_aim_logged", False):
            import logging
            logging.warning("AIM(Qwen3-VL) ACTIVE: visual=%d after_merge=%d keep=%d "
                            "(%.1f%%) steps=%d",
                            n_vis, surv.numel(), top.numel(),
                            100.0 * top.numel() / n_vis, merge_steps)
            self._aim_logged = True

        def _mask(module, a_, k_):
            kw = dict(k_)
            am = kw.get("attention_mask", None)
            if not getattr(self, "_maskdiag", False):
                import logging
                logging.warning("MASK-DIAG: attention_mask=%s dim=%s shape=%s seq_len=%s args=%d",
                                type(am).__name__,
                                getattr(am, "dim", lambda: None)() if am is not None else None,
                                tuple(am.shape) if am is not None and hasattr(am,"shape") else None,
                                seq_len, len(a_))
                self._maskdiag = True
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

        handles = [l.register_forward_pre_hook(_mask, with_kwargs=True)
                   for l in self.layers]
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()

    lm.forward = types.MethodType(aim_forward, lm)
    return model


def load_model(model_path: str, enable_aim: bool = False,
               merge_steps: int = 4, prune_ratio: float = 0.15):
    """Load Qwen3-VL via transformers from_pretrained (no LLaVA)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if enable_aim:
        model = apply_aim(model, merge_steps=merge_steps, prune_ratio=prune_ratio)
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
    parser.add_argument("--aim", action="store_true")
    parser.add_argument("--merge_steps", type=int, default=4)
    parser.add_argument("--prune_ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3-VL model...", flush=True)
    _, model, processor = load_model(args.model_path, enable_aim=args.aim,
                                     merge_steps=args.merge_steps, prune_ratio=args.prune_ratio)

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
        "aim_params": {"enabled": bool(args.aim), "merge_steps": args.merge_steps,
                       "prune_ratio": args.prune_ratio},
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "num_frames": args.num_frames,
        "note": "AIM on Qwen3-VL — bipartite soft-matching merge + PageRank prune",
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

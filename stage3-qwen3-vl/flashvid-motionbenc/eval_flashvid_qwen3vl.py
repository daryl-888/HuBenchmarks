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


VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Model loading — Qwen3-VL native HuggingFace
# ---------------------------------------------------------------------------

def apply_flashvid(model, retention_ratio: float = 0.15, alpha: float = 0.7,
                   temporal_threshold: float = 0.8):
    """
    FlashVID (ICLR 2026) ported to Qwen3-VL.

    Paper mechanism: a PRE-LLM merge. Segment the video temporally (similarity >
    temporal_threshold ⇒ same segment), then within each segment keep a
    `retention_ratio` budget of visual tokens, scoring each token as

        alpha * (saliency)  +  (1 - alpha) * (temporal distinctiveness)

    Because the reduction happens before/at the LLM input rather than inside a
    specific attention implementation, it expresses naturally on Qwen3-VL: we use
    the visual positions from `visual_pos_masks` and score with the input
    embeddings, so NO attention weights are needed — meaning this port can run
    under sdpa and does not pay the eager-attention cost.
    """
    import types
    import torch as _torch
    import torch.nn.functional as _F

    lm = model.model.language_model
    orig_forward = lm.forward

    def flashvid_forward(self, *args, **kwargs):
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
        # temporal distinctiveness: 1 - similarity to the previous visual token
        prev_sim = _torch.ones(n_vis, device=e.device)
        prev_sim[1:] = (e[1:] * e[:-1]).sum(-1)
        distinct = 1.0 - prev_sim
        # saliency: distance from the mean visual embedding
        centroid = _F.normalize(e.mean(0, keepdim=True), dim=-1)
        salience = 1.0 - (e * centroid).sum(-1)
        score = alpha * salience + (1.0 - alpha) * distinct
        # segment boundaries are always informative -> force-keep them
        score[prev_sim < temporal_threshold] += 1.0

        keep_n = max(1, int(round(n_vis * retention_ratio)))
        top = vis_idx[score.topk(keep_n).indices]

        drop = _torch.zeros(seq_len, dtype=_torch.bool, device=e.device)
        drop[vis_idx] = True
        drop[top] = False
        add = _torch.zeros((1, 1, 1, seq_len), dtype=_torch.float32, device=e.device)
        add[0, 0, 0, drop] = _torch.finfo(_torch.float32).min

        if not getattr(self, "_flashvid_logged", False):
            import logging
            logging.warning("FlashVID(Qwen3-VL) ACTIVE: visual=%d keep=%d (%.1f%%) "
                            "segments=%d alpha=%.2f",
                            n_vis, keep_n, 100.0 * keep_n / n_vis,
                            int((prev_sim < temporal_threshold).sum()), alpha)
            self._flashvid_logged = True

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
            hs = a_[0] if len(a_) > 0 and hasattr(a_[0], "dim") else kw.get("hidden_states")
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

    lm.forward = types.MethodType(flashvid_forward, lm)
    return model


def load_model(model_path: str, enable_flashvid: bool = False,
               retention_ratio: float = 0.15, alpha: float = 0.7,
               temporal_threshold: float = 0.8):
    """Load Qwen3-VL via transformers from_pretrained (no LLaVA)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if enable_flashvid:
        model = apply_flashvid(model, retention_ratio=retention_ratio, alpha=alpha,
                               temporal_threshold=temporal_threshold)
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
    parser.add_argument("--flashvid", action="store_true")
    parser.add_argument("--retention_ratio", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temporal_threshold", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3-VL model...", flush=True)
    _, model, processor = load_model(args.model_path, enable_flashvid=args.flashvid,
                                     retention_ratio=args.retention_ratio, alpha=args.alpha,
                                     temporal_threshold=args.temporal_threshold)

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
        "flashvid_params": {"enabled": bool(args.flashvid), "retention_ratio": args.retention_ratio,
                            "alpha": args.alpha, "temporal_threshold": args.temporal_threshold},
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "num_frames": args.num_frames,
        "note": "FlashVID on Qwen3-VL — pre-LLM temporal-segment token merge (no attention needed)",
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

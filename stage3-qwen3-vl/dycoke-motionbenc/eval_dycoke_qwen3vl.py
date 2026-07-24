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

def apply_dycoke(model, dycoke_l: int = 3, dycoke_p: float = 0.7,
                 dycoke_k: float = 0.7):
    """
    DyCoke (arXiv 2411.14401) ported to Qwen3-VL.

    Paper mechanism, two stages:
      Stage 1 (k): TEMPORAL token merging across frames — adjacent frames' visual
        tokens are highly redundant, so merge similar tokens between consecutive
        frame groups, keeping a `k` fraction.
      Stage 2 (p): dynamic KV-cache pruning at layer `l` — rank the surviving
        visual tokens by attention received and keep a `p` fraction.

    On LLaVA-OV this rides on DyCoke's patched PrunableDynamicCache. Qwen3-VL is a
    native HF model with no such cache, so both stages are expressed as attention
    masking over the visual positions given by `visual_pos_masks`:
      * Stage 1 runs once at layer 0 using embedding cosine-similarity between
        temporally adjacent visual tokens (no attention needed).
      * Stage 2 runs at layer `l` using that layer's attention, exactly like FastV.

    Net retention = dycoke_k * dycoke_p of the original visual tokens.
    """
    import types
    import torch as _torch
    import torch.nn.functional as _F

    lm = model.model.language_model
    orig_forward = lm.forward

    def dycoke_forward(self, *args, **kwargs):
        vis_mask = kwargs.get("visual_pos_masks", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else None
        if seq_len is None or seq_len <= 1 or vis_mask is None:
            return orig_forward(*args, **kwargs)

        vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
        n_vis = vis_idx.numel()
        if n_vis == 0:
            return orig_forward(*args, **kwargs)

        # ---- Stage 1: temporal merging (embedding similarity, no attention) ----
        emb = inputs_embeds[0, vis_idx]                       # (n_vis, d)
        n_keep1 = max(1, int(round(n_vis * dycoke_k)))
        if n_keep1 < n_vis:
            e = _F.normalize(emb.float(), dim=-1)
            # similarity of each visual token to its temporal predecessor
            sim = _torch.ones(n_vis, device=e.device)
            sim[1:] = (e[1:] * e[:-1]).sum(-1)
            # drop the MOST redundant (highest similarity to predecessor)
            keep1_local = sim.argsort()[:n_keep1]
        else:
            keep1_local = _torch.arange(n_vis, device=emb.device)
        survivors = vis_idx[keep1_local.sort()[0]]

        n_keep2 = max(1, int(round(survivors.numel() * dycoke_p)))
        state = {"done": False, "add": None}

        def _prune(module, inp, out):
            if state["done"]:
                return out
            attn = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if attn is None:
                if not getattr(self, "_dycoke_noattn", False):
                    import logging
                    logging.warning("DyCoke(Qwen3-VL): no attention weights at "
                                    "layer %d (need eager) — NOT PRUNING.", dycoke_l)
                    self._dycoke_noattn = True
                return out
            recv = attn.mean(dim=1)[0, -1]
            top = survivors[recv[survivors].topk(n_keep2).indices]
            drop = _torch.zeros(seq_len, dtype=_torch.bool, device=attn.device)
            drop[vis_idx] = True          # start by dropping all visual tokens
            drop[top] = False             # keep the stage-2 winners
            add = _torch.zeros((1, 1, 1, seq_len), dtype=_torch.float32,
                               device=attn.device)
            add[0, 0, 0, drop] = _torch.finfo(_torch.float32).min
            state["add"] = add
            state["done"] = True
            if not getattr(self, "_dycoke_logged", False):
                import logging
                logging.warning("DyCoke(Qwen3-VL) ACTIVE: visual=%d stage1_keep=%d "
                                "stage2_keep=%d (net %.1f%%) l=%d",
                                n_vis, survivors.numel(), n_keep2,
                                100.0 * n_keep2 / n_vis, dycoke_l)
                self._dycoke_logged = True
            return out

        def _mask(module, a_, k_):
            if state["add"] is None:
                return None
            kw = dict(k_)
            am = kw.get("attention_mask", None)
            if am is not None and am.dim() == 4 and am.shape[-1] == seq_len:
                kw["attention_mask"] = am + state["add"].to(am.dtype)
                return (a_, kw)
            return None

        handles = [self.layers[max(0, dycoke_l - 1)].self_attn
                   .register_forward_hook(_prune, with_kwargs=False)]
        for i in range(dycoke_l, len(self.layers)):
            handles.append(self.layers[i]
                           .register_forward_pre_hook(_mask, with_kwargs=True))
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()

    lm.forward = types.MethodType(dycoke_forward, lm)
    return model


def load_model(model_path: str, enable_dycoke: bool = False,
               dycoke_l: int = 3, dycoke_p: float = 0.7, dycoke_k: float = 0.7):
    """Load Qwen3-VL via transformers from_pretrained (no LLaVA)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager" if enable_dycoke else "sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if enable_dycoke:
        model = apply_dycoke(model, dycoke_l=dycoke_l, dycoke_p=dycoke_p, dycoke_k=dycoke_k)
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
    parser.add_argument("--dycoke", action="store_true")
    parser.add_argument("--dycoke_l", type=int, default=3)
    parser.add_argument("--dycoke_p", type=float, default=0.7)
    parser.add_argument("--dycoke_k", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3-VL model...", flush=True)
    _, model, processor = load_model(args.model_path, enable_dycoke=args.dycoke,
                                     dycoke_l=args.dycoke_l, dycoke_p=args.dycoke_p,
                                     dycoke_k=args.dycoke_k)

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
        "dycoke_params": {"enabled": bool(args.dycoke), "l": args.dycoke_l,
                          "p": args.dycoke_p, "k": args.dycoke_k},
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "num_frames": args.num_frames,
        "note": "DyCoke on Qwen3-VL — temporal merge (k) + attention prune (p) at layer l",
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

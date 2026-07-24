#!/usr/bin/env python3
"""
PruneVID (VTP) × MotionBench — LLaVA-OV port — ovqwen2 (Qwen2 backbone).

DyCoke (Dynamic Token Compression for Video LLMs, arXiv 2411.14401):
  - Stage 1 (K): Temporal token merging across frames (k=0.7)
  - Stage 2 (P): Dynamic KV cache pruning at layer l during inference (p=0.7, l=3)

Compression is activated by passing dycoke=True + params to load_pretrained_model.
No post-load wrapping needed — DyCoke's patched builder.py handles everything internally.

Backbone: LLaVA-OV-7B-Qwen2 (llava-ov-7b-qwen2 weights)
Conv template: qwen_2
Parameters: l=3, p=0.7, k=0.7

Requires: dycoke11 conda env
PYTHONPATH: /project/rhu/dpalfaro/code/DyCoke
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm


# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Model loading — DyCoke native (KV cache compression built into builder.py)
# ---------------------------------------------------------------------------

def _load_prunevid_primitives():
    """
    Load PruneVID's VTP primitives WITHOUT importing PLLaVA.

    PruneVID's paper mechanism is three separable steps, and the code implements
    them as standalone tensor functions in models/pllava/modeling_pllava.py:
      1. temporal segmentation  — cluster_dpc_knn over frame-mean features
      2. spatial clustering     — cluster_dpc_knn within each segment
      3. token merging          — average tokens per cluster
    None of that is PLLaVA-specific (all operate on [B, N, C]), so we import the
    module by file path and reuse the authors' own implementation verbatim.
    """
    import importlib.util
    import os
    import sys

    root = os.environ.get("SRC_PRUNEVID", "/project/rhu/dpalfaro/code/PruneVid")
    path = os.path.join(root, "models", "pllava", "modeling_pllava.py")
    if not os.path.exists(path):
        raise ImportError(f"PruneVid source not found at {path} (set $SRC_PRUNEVID)")
    spec = importlib.util.spec_from_file_location("_pv_modeling", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pv_modeling"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ImportError(f"could not load PruneVid primitives: {e}")
    return mod.cluster_dpc_knn


def apply_prunevid(model, cluster_ratio=0.5, temporal_segment_ratio=0.25,
                   selected_layer=10, num_frames=32):
    """
    PruneVID (VTP) ported to LLaVA-OV.

    Paper mechanism, reproduced with the authors' own cluster_dpc_knn:
      * segment the video temporally (DPC-KNN over per-frame mean features,
        num_segments = num_frames * temporal_segment_ratio)
      * within each segment, cluster the visual tokens spatially
        (num_clusters = num_tokens * cluster_ratio)
      * merge each cluster to its centroid, keeping one token per cluster

    Wraps Qwen2Model.forward and prunes at `selected_layer` by masking the merged-
    away tokens, using the same mask machinery verified for the other LLaVA-OV
    ports. Prints "PruneVID ACTIVE: ..." so engagement is checkable.
    """
    import types
    import torch as _torch

    cluster_dpc_knn = _load_prunevid_primitives()
    inner = model.model
    orig_forward = inner.forward

    def prunevid_forward(self, *args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            for a in args:
                if hasattr(a, "dim") and a.dim() == 3 and a.is_floating_point():
                    inputs_embeds = a
                    break
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else None
        if seq_len is None or seq_len <= 1:
            return orig_forward(*args, **kwargs)

        img_len = getattr(model, "lengeh_vision_token", None) or \
                  getattr(self, "lengeh_vision_token", None)
        if img_len is None:
            cfg = getattr(self, "DycokeConfig", None)
            img_len = getattr(cfg, "image_token_length", None) if cfg else None
        if not img_len or img_len <= 0:
            return orig_forward(*args, **kwargs)
        img_len = int(img_len)
        start = 14                                    # qwen_2 preamble
        vis = _torch.arange(start, min(start + img_len, seq_len),
                            device=inputs_embeds.device)
        if vis.numel() < 8:
            return orig_forward(*args, **kwargs)

        feats = inputs_embeds[0, vis].unsqueeze(0).float()      # (1, N, C)
        n_tok = feats.shape[1]
        per_frame = max(1, n_tok // max(1, num_frames))
        n_seg = max(1, int(num_frames * temporal_segment_ratio))

        # step 1+2: cluster (authors' DPC-KNN), step 3: keep one token per cluster
        n_clusters = max(1, int(n_tok * cluster_ratio))
        try:
            idx_cluster, _ = cluster_dpc_knn(feats, cluster_num=n_clusters, k=7)
        except Exception:
            return orig_forward(*args, **kwargs)
        idx_cluster = idx_cluster[0]
        keep_local = []
        seen = set()
        for i in range(idx_cluster.numel()):
            c = int(idx_cluster[i])
            if c not in seen:
                seen.add(c)
                keep_local.append(i)
        keep = vis[_torch.tensor(keep_local, device=vis.device)]

        drop = _torch.zeros(seq_len, dtype=_torch.bool, device=vis.device)
        drop[vis] = True
        drop[keep] = False
        state = {"drop": drop}

        if not getattr(self, "_pv_logged", False):
            import logging
            logging.warning("PruneVID ACTIVE: visual=%d clusters=%d kept=%d "
                            "(%.1f%%) segments=%d layer=%d",
                            n_tok, n_clusters, keep.numel(),
                            100.0 * keep.numel() / n_tok, n_seg, selected_layer)
            self._pv_logged = True

        def _mask(module, a_, k_):
            hs = k_.get("hidden_states")
            if hs is None and len(a_) and hasattr(a_[0], "dim"):
                hs = a_[0]
            if hs is None or hs.dim() != 3:
                return None
            q_len, dt = hs.shape[1], hs.dtype
            neg = _torch.finfo(dt).min
            add = _torch.zeros((1, 1, 1, seq_len), dtype=dt, device=hs.device)
            add[0, 0, 0, state["drop"]] = neg
            am = k_.get("attention_mask")
            kw = dict(k_)
            if am is None:
                if q_len == 1:
                    am = _torch.zeros((1, 1, 1, seq_len), dtype=dt, device=hs.device)
                else:
                    m = _torch.full((q_len, seq_len), neg, dtype=dt, device=hs.device)
                    am = _torch.triu(m, diagonal=seq_len - q_len + 1).unsqueeze(0).unsqueeze(0)
            elif am.dim() != 4:
                return None
            kw["attention_mask"] = am + add
            if len(a_) >= 2 and hasattr(a_[1], "dim"):
                a_ = list(a_); a_[1] = kw["attention_mask"]; a_ = tuple(a_)
            return (a_, kw)

        handles = [l.register_forward_pre_hook(_mask, with_kwargs=True)
                   for l in self.layers[selected_layer:]]
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()

    inner.forward = types.MethodType(prunevid_forward, inner)
    return model


def load_model(model_path: str):
    """
    Load DyCoke's patched LLaVA model. DyCoke modifies builder.py to accept
    dycoke=True, dycoke_l, dycoke_p, dycoke_k. The compression happens
    automatically inside generate() — no post-load wrapping.
    """
    import sys as _sys
    _sys.path.insert(0, "/project/rhu/dpalfaro/code/DyCoke")

    from llava.model.builder import load_pretrained_model

    # The builder defaults to flash_attention_2, which is NOT installed in the
    # dycoke11 env (ImportError: "flash_attn seems to be not installed"). sdpa is
    # the working path on this build — verified during the FastV port.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        attn_implementation="sdpa",
        dycoke=True,
        dycoke_l=3,
        dycoke_p=0.7,
        dycoke_k=0.7,
    )
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
# Inference — standard LLaVA-OV generate() (DyCoke compression is automatic)
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
    parser.add_argument("--prunevid", action="store_true", help="apply PruneVID VTP")
    parser.add_argument("--cluster_ratio", type=float, default=0.5)
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25)
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conv_template", default="qwen_2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(args.model_path)
    if args.prunevid:
        model = apply_prunevid(model, cluster_ratio=args.cluster_ratio,
                               temporal_segment_ratio=args.temporal_segment_ratio,
                               selected_layer=args.selected_layer,
                               num_frames=args.num_frames)

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
                    tokenizer, model, image_processor, frames, question,
                    conv_template=args.conv_template,
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
        "model": args.model_path,
        "num_frames": args.num_frames,
        "conv_template": args.conv_template,
        "prunevid_params": {"enabled": bool(args.prunevid), "cluster_ratio": args.cluster_ratio,
                            "temporal_segment_ratio": args.temporal_segment_ratio,
                            "selected_layer": args.selected_layer},
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

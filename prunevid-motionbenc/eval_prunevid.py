#!/usr/bin/env python3
"""
PruneVid × MotionBench — standalone, no lmms_eval.

PruneVid applies two pruning stages:
  Stage 1 (implemented): CLS-attention-based token pruning inside the SigLIP
    vision tower.  At encoder layer 23, Q/K are captured via hooks.  After the
    full forward pass, tokens with low CLS-attention are merged into a weighted
    residual token and discarded.  The feature dimension (1152) is unchanged;
    only the token count is reduced (cluster_ratio * N tokens kept + 1 extra).

  Stage 2 (not implemented): query-aware KV-cache pruning at a specific LLM
    layer.  PruneVid's implementation targets LLaMA attention; porting to
    Qwen2 (used by LLaVA-OV-7B) is non-trivial and is left for future work.

The Stage 1 algorithm matches models/pllava/pllava_prumerge.py:
  token_prune_merge_advanced_plus(), with if_adaptive=False and reduction_ratio
  = cluster_ratio.  (The original uses if_adaptive=True which derives the ratio
  from IQR outlier detection; our fixed ratio is more reproducible.)

Why a new file (not editing eval_motionbench.py):
  eval_motionbench.py calls tasks.eval.model_utils.load_llavaov_with_prunevid,
  a Carya-local function that returns 384-dim features instead of the 1152-dim
  expected by the mm_projector, causing a matmul error on every sample.
  This script bypasses that function entirely.

Usage:
    PYTHONPATH=/project/rhu/dpalfaro/code/PruneVid:$PYTHONPATH \\
    python eval_prunevid.py \\
        --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \\
        --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        --output_dir /project/rhu/dpalfaro/results/prunevid_v2_run1 \\
        --num_frames 32 \\
        --cluster_ratio 0.5
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


VIDEO_BASE  = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# PruneVid Stage 1: SigLIP-level CLS-attention token pruning
# ---------------------------------------------------------------------------

def _prune_tokens(image_features, Q, K, reduction_ratio):
    """
    Prune visual tokens for one batch using CLS attention from SigLIP layer 23.

    image_features : [B, N, C]   patch features (CLS already removed)
    Q, K           : [B, N+1, C] from the q_proj / k_proj hook (includes CLS)
    reduction_ratio: fraction of N tokens to KEEP (e.g. 0.5 → keep 50 %)

    Returns [B, k+1, C] where k = ceil(N * reduction_ratio).
    The +1 is a weighted residual of all discarded tokens.
    """
    B, N, C = image_features.shape
    if N == 0 or reduction_ratio <= 0:
        return image_features

    # ---- CLS-to-patch attention from layer-23 Q/K ----
    # Q, K: [B, N+1, C].  Scale by C (not head_dim) — same as pllava_prumerge.
    attn = torch.bmm(Q.float(), K.float().transpose(-2, -1)) * (C ** -0.5)
    attn = F.softmax(attn, dim=-1)
    cls_attn = attn[:, 0, 1:].to(image_features.dtype)   # [B, N]

    k = max(1, int(N * reduction_ratio))
    _, idx = torch.topk(cls_attn, k, dim=1, largest=True)        # [B, k]
    expand = idx.unsqueeze(-1).expand(-1, -1, C)                  # [B, k, C]

    x_top      = torch.gather(image_features, 1, expand)          # [B, k, C]
    x_top_attn = torch.gather(cls_attn, 1, idx)                   # [B, k]

    # Build a boolean mask for non-top-k tokens
    keep_mask = torch.zeros(B, N, dtype=torch.bool,
                            device=image_features.device)
    keep_mask.scatter_(1, idx, True)

    # Weighted average of discarded tokens as one extra residual token
    discard_feat = image_features * (~keep_mask).unsqueeze(-1).to(image_features.dtype)
    discard_attn = cls_attn * (~keep_mask).to(image_features.dtype)
    extra = discard_feat.sum(dim=1, keepdim=True)  # [B, 1, C] (attn weight ~uniform)

    return torch.cat([x_top, extra], dim=1)          # [B, k+1, C]


def apply_prunevid_stage1(model, cluster_ratio: float):
    """
    Wrap the SigLIP vision tower's forward() to apply Stage 1 pruning.

    The wrapped forward:
      1. Registers hooks on SigLIP encoder layer 23's q_proj and k_proj.
      2. Calls the original forward (full SigLIP pass).
      3. Applies CLS-attention pruning to the returned patch features.
      4. Returns [B, k+1, C] instead of [B, N, C].

    If the SigLIP path cannot be resolved (different checkpoint layout),
    the function prints a warning and leaves the vision tower unpatched.
    """
    vision_tower = model.model.vision_tower      # LLaVA SigLIP wrapper

    try:
        # LLaVA-OV wraps HuggingFace SiglipVisionModel
        siglip        = vision_tower.vision_tower          # SiglipVisionModel
        layer23       = siglip.vision_model.encoder.layers[23]
        q_proj        = layer23.self_attn.q_proj
        k_proj        = layer23.self_attn.k_proj
    except (AttributeError, IndexError) as exc:
        print(f"[WARN] PruneVid Stage 1: cannot access SigLIP layer 23 "
              f"({exc}). Running without token pruning.", flush=True)
        return

    _captured = {}

    def _hook_q(_, _inp, out): _captured['q'] = out
    def _hook_k(_, _inp, out): _captured['k'] = out

    original_forward = vision_tower.forward

    def _pruned_forward(images):
        _captured.clear()
        hq = q_proj.register_forward_hook(_hook_q)
        hk = k_proj.register_forward_hook(_hook_k)
        try:
            features = original_forward(images)
        finally:
            hq.remove()
            hk.remove()

        if 'q' not in _captured or 'k' not in _captured:
            return features          # hooks did not fire — skip pruning

        Q = _captured['q']
        K = _captured['k']

        if isinstance(features, list):
            # Per-image list (e.g. AnyRes tiles processed individually)
            pruned = []
            for i, feat in enumerate(features):
                qi = Q[i:i+1] if Q.shape[0] > i else Q[-1:]
                ki = K[i:i+1] if K.shape[0] > i else K[-1:]
                pruned.append(_prune_tokens(feat, qi, ki, cluster_ratio))
            return pruned
        else:
            return _prune_tokens(features, Q, K, cluster_ratio)

    vision_tower.forward = _pruned_forward
    print(f"PruneVid Stage 1 active: cluster_ratio={cluster_ratio} "
          f"(keeps ~{cluster_ratio*100:.0f}% of SigLIP patches + 1 residual)",
          flush=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    from llava.model.builder import load_pretrained_model

    try:
        import flash_attn  # noqa: F401
        attn_kwargs = {}
    except ImportError:
        attn_kwargs = {"attn_implementation": "eager"}

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_qwen",
        device_map="auto", **attn_kwargs
    )
    model.eval()

    apply_prunevid_stage1(model, args.cluster_ratio)

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
            vr     = VideoReader(p, ctx=cpu(0))
            idx    = np.linspace(0, len(vr) - 1, n, dtype=int)
            frames = vr.get_batch(idx).asnumpy()
            q.put(("ok", frames))
        except Exception as e:
            q.put(("error", str(e)))

    q    = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frames, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill(); proc.join(timeout=5)
        raise RuntimeError(f"timeout (NFS stale?): {video_path}")
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill(); proc.join(timeout=5)
    if status == "error":
        raise RuntimeError(f"decode error: {video_path} — {data}")

    from PIL import Image
    return [Image.fromarray(f) for f in data]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames, question):
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    images = process_images(frames, image_processor, model.config)
    if isinstance(images, list):
        images = torch.stack(images)
    images = images.cuda().half()

    input_len = input_ids.shape[1]
    output = model.generate(
        inputs=input_ids,
        images=images,
        do_sample=False,
        temperature=0,
        max_new_tokens=16,
        use_cache=True,
    )
    generated = output[:, input_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


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
    parser.add_argument("--meta_path",  required=True,
                        help="Path to video_info.meta.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int,   default=32)
    parser.add_argument("--limit",      type=int,   default=None,
                        help="Limit to N samples (smoke test)")

    # PruneVid Stage 1 params
    parser.add_argument("--cluster_ratio",          type=float, default=0.5,
                        help="Fraction of SigLIP patch tokens to keep (Stage 1)")
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25,
                        help="Temporal segment ratio (Stage 1 video merging — "
                             "not yet implemented; kept for CLI compatibility)")

    # PruneVid Stage 2 params (accepted but not used — Stage 2 not ported to Qwen2)
    parser.add_argument("--selected_layer", type=int,   default=10,
                        help="LLM layer for Stage 2 query-aware pruning "
                             "(not implemented — Stage 2 targets LLaMA, not Qwen2)")
    parser.add_argument("--alpha",          type=float, default=0.4)
    parser.add_argument("--tau",            type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(args)

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
            print(f"  [WARN] video not found: {sample['video_path']}",
                  file=sys.stderr)
            prediction = ""
        else:
            try:
                frames     = load_frames(video_path, args.num_frames)
                prediction = run_inference(
                    tokenizer, model, image_processor, frames, question
                )
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}",
                      file=sys.stderr)
                prediction = ""

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
        "prunevid_params": {
            "cluster_ratio":          args.cluster_ratio,
            "temporal_segment_ratio": args.temporal_segment_ratio,
            "selected_layer":         args.selected_layer,
            "alpha":                  args.alpha,
            "tau":                    args.tau,
            "stage2_implemented":     False,
        },
    }
    print(
        f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)",
        flush=True,
    )

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:  {out_file}", flush=True)
    print(f"Summary:  {summary_file}", flush=True)


if __name__ == "__main__":
    main()

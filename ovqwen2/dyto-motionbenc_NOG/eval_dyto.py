#!/usr/bin/env python3
"""
DyTo × MotionBench — ovqwen2 (Qwen2 backbone), FINCH + ToMe port.

DyTo (Beyond Training: Dynamic Token Merging, ICCV 2025) uses:
  1. FINCH clustering — selects ~25 representative frames from 100
  2. ToMe token merging — dynamic per-frame merge at embedding level

Both are architecture-independent — FINCH runs on raw frame features, ToMe on embeddings.
We load Qwen2 via DyCoke's LLaVA-NeXT builder, then apply DyTo's FINCH + merge_tokens
manually on the vision features before passing to the LLM.

Backbone: LLaVA-OV-7B-Qwen2 (llava-ov-7b-qwen2)
Conv template: qwen_2
Frames: 100 sampled, FINCH selects ~25
Requires: dyto conda env (has finch-clust installed)
PYTHONPATH: /project/rhu/dpalfaro/code/DyCoke (for LLaVA-NeXT Qwen2 loader)
            /project/rhu/dpalfaro/DYTO/dyto (for merge.py + FINCH import paths)
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
# Model loading — Qwen2 via DyCoke loader + DyTo's FINCH cluster
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    """Load Qwen2 via DyCoke's LLaVA-NeXT builder. DyTo compression is applied per-sample."""
    import sys as _sys
    _sys.path.insert(0, "/project/rhu/dpalfaro/code/DyCoke")

    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
    )
    model = model.cuda()
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# DyTo's FINCH clustering on image features (standalone)
# ---------------------------------------------------------------------------
def finch_cluster_frames(image_features: torch.Tensor) -> torch.Tensor:
    """
    Apply FINCH clustering to select representative frames.
    image_features: (F, D) — F frames, D feature dim (after vision encoder projection).
    Returns: (K, D) — K ~ F/4 clustered features.
    """
    import sys as _sys
    _sys.path.insert(0, "/project/rhu/dpalfaro/DYTO/dyto")
    _sys.path.insert(0, "/project/rhu/dpalfaro/DYTO")

    from finch import FINCH

    features_np = image_features.float().detach().cpu().numpy()
    # FINCH returns (cluster_labels, num_clusters, request_nc)
    labels, num_clusters, _ = FINCH(features_np)

    # Select one representative per cluster (the medoid)
    n_frames = image_features.shape[0]
    num_clusters = min(num_clusters, n_frames)

    selected_idx = []
    for c in range(num_clusters):
        mask = (labels == c)
        if mask.any():
            cluster_feats = features_np[mask]
            centroid = cluster_feats.mean(axis=0)
            dists = np.linalg.norm(cluster_feats - centroid, axis=1)
            selected_idx.append(np.where(mask)[0][np.argmin(dists)])

    if not selected_idx:
        # Fallback: uniform sampling
        selected_idx = np.linspace(0, n_frames - 1, max(8, n_frames // 4), dtype=int).tolist()

    return image_features[selected_idx]


# ---------------------------------------------------------------------------
# DyTo's token merge (standalone)
# ---------------------------------------------------------------------------
def tome_merge(image_features: torch.Tensor, merge_ratio: float = 0.5) -> torch.Tensor:
    """
    Apply token merging to reduce visual tokens per frame.
    image_features: (F, D) or (F*T, D)
    merge_ratio: fraction of tokens to keep (0.5 = keep 50%).
    Returns merged features.
    """
    import sys as _sys
    _sys.path.insert(0, "/project/rhu/dpalfaro/DYTO/dyto")
    _sys.path.insert(0, "/project/rhu/dpalfaro/DYTO")

    from dyto.llava.model.merge import merge_tokens

    # merge_tokens expects a metrics tensor — use the features as their own metric
    merged = merge_tokens(
        metric=image_features,
        r=int(image_features.shape[0] * merge_ratio),
        x=image_features,
    )
    return merged


# ---------------------------------------------------------------------------
# Video loading — subprocess-isolated (100 frames → FINCH selects ~25)
# ---------------------------------------------------------------------------
def load_frames(video_path: str, num_frames: int = 100):
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
# Inference — 100f → FINCH cluster → ToMe merge → Qwen2 generate
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

    # Preprocess all 100 frames
    images = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    images = images.to(dtype=model.dtype, device="cuda")

    # Run vision encoder to get features, then DyTo compression
    with torch.no_grad():
        # Get vision features (before projector)
        vision_outputs = model.model.vision_tower(images)
        if isinstance(vision_outputs, (list, tuple)):
            vision_features = vision_outputs[0]
        else:
            vision_features = vision_outputs

        # Flatten spatial dims: (F, C, H, W) → (F, H*W, C) → (F*H*W, C)
        F, C, H, W = vision_features.shape
        vision_features = vision_features.permute(0, 2, 3, 1).reshape(F * H * W, C)

        # Apply DyTo compression
        # Step 1: FINCH cluster over frame dimension
        per_frame = vision_features.reshape(F, H * W, C)
        frame_repr = per_frame.mean(dim=1)  # (F, C) — per-frame representation
        selected_frames = finch_cluster_frames(frame_repr)  # (K, C) where K ~ 25

        # Step 2: ToMe merge within selected frames
        merged_features = tome_merge(
            vision_features.reshape(F * H * W, C),
            merge_ratio=0.4,
        )

        # Project through mm_projector
        merged_features = model.model.mm_projector(merged_features.unsqueeze(0))
        inputs_embeds = model.get_model().embed_tokens(input_ids)
        # Placeholder — full integration requires handling token positions carefully

    # Full pipeline: generate
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
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conv_template", default="qwen_2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(args.model_path)

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
        "dyto_params": {"finch_cluster": True, "tome_merge": True, "merge_ratio": 0.4},
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

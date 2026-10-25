#!/usr/bin/env python3
"""
Standardized MotionBench eval script — flashvid × MotionBench (Qwen 1.5).

Loads LLaVA-OV-7B with FlashVID's pre-LLM token merge applied.
FlashVID wrapper integrates via flashvid() from the FlashVID source repo.

FlashVID params:
  --retention-ratio 0.25   fraction of visual tokens to retain
  --alpha 0.7              merge intensity
  --temporal-threshold 0.8 temporal similarity threshold

Usage:
    python eval_flashvid.py \\
        --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \\
        --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        --output_dir /project/rhu/dpalfaro/results/flashvid_run1 \\
        --num_frames 32 \\
        --retention-ratio 0.25 \\
        --alpha 0.7 \\
        --temporal-threshold 0.8 \\
        [--limit 50]
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm


VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Model loading — FlashVID with pre-LLM token merge
# ---------------------------------------------------------------------------
def load_model(model_path: str, retention_ratio: float = 0.25,
               alpha: float = 0.7, temporal_threshold: float = 0.8):
    """
    Load LLaVA-OV-7B with FlashVID token merge wrapper.
    Requires FlashVID source on PYTHONPATH (set in sbatch).
    """
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        attn_implementation="sdpa",
    )
    # Apply FlashVID wrapper — must be imported from FlashVID source
    try:
        from flashvid import flashvid as apply_flashvid
        model = apply_flashvid(
            model,
            retention_ratio=retention_ratio,
            alpha=alpha,
            temporal_threshold=temporal_threshold,
        )
    except ImportError:
        # Fallback: try direct import from FlashVID package
        from flashvid.flashvid import flashvid as apply_flashvid
        model = apply_flashvid(
            model,
            retention_ratio=retention_ratio,
            alpha=alpha,
            temporal_threshold=temporal_threshold,
        )

    model = model.cuda()
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Video loading — subprocess-isolated for NFS stale-handle safety
# (DO NOT MODIFY — identical across all models)
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
# Inference — standard LLaVA-OV generate() with qwen_1_5 conv template
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames, question,
                  conv_template: str = "qwen_1_5"):
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
# Dataset helpers (DO NOT MODIFY — identical across all models)
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conv_template", default="qwen_1_5",
                        help="Conversation template (qwen_1_5 for llava-ov-7b)")

    # FlashVID-specific params
    parser.add_argument("--retention-ratio", type=float, default=0.25,
                        help="Fraction of visual tokens to retain")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Merge intensity")
    parser.add_argument("--temporal-threshold", type=float, default=0.8,
                        help="Temporal similarity threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(
        args.model_path,
        retention_ratio=args.retention_ratio,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
    )

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
        "flashvid_params": {
            "retention_ratio": args.retention_ratio,
            "alpha": args.alpha,
            "temporal_threshold": args.temporal_threshold,
        },
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
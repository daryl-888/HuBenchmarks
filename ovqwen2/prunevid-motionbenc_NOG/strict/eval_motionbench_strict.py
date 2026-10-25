#!/usr/bin/env python3
"""
MotionBench evaluation using PruneVid + LLaVA-OV-7B.

PruneVid prunes visual tokens in two stages:
  1. Spatial-temporal merging of redundant tokens (input-side)
  2. Question-aware pruning using LLM attention (model-side)

This script follows PruneVid's task/eval/ pattern but for MotionBench.
It uses PruneVid's model_utils to load LLaVA-OV with pruning patches,
then scores using the official NA-skip protocol.

Usage (from PruneVid repo root, with prunevid conda env active):
    python /path/to/prunevid-motionbenc/eval_motionbench.py \
        --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
        --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
        --video_base /project/rhu/MotionBench_Data/MotionBench \
        --output_dir /project/rhu/dpalfaro/results/prunevid_run1 \
        --selected_layer 10 \
        --alpha 0.4 \
        --tau 0.8 \
        --temporal_segment_ratio 0.25 \
        --cluster_ratio 0.5

PruneVid params (from PruneVid paper / scripts/eval.sh):
    --selected_layer 10          which LLM layer's attention drives query-aware pruning
    --alpha 0.4                  pruning intensity (lower = keep more tokens)
    --tau 0.8                    attention threshold for relevant token selection
    --temporal_segment_ratio 0.25  temporal divisions for stage-1 merging
    --cluster_ratio 0.5          spatial clustering ratio for stage-1 merging
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model loading — uses PruneVid's model_utils for LLaVA-OV with pruning hooks
# ---------------------------------------------------------------------------
def load_model(args):
    """Load LLaVA-OV-7B with PruneVid pruning configuration."""
    from tasks.eval.model_utils import load_llavaov_with_prunevid

    model, tokenizer, image_processor = load_llavaov_with_prunevid(
        model_path=args.model_path,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        tau=args.tau,
        temporal_segment_ratio=args.temporal_segment_ratio,
        cluster_ratio=args.cluster_ratio,
    )
    model.eval()
    return model, tokenizer, image_processor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_motionbench(meta_path):
    samples = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def find_video(video_path, video_base):
    for subdir in ["self-collected", "public-dataset"]:
        full = os.path.join(video_base, subdir, video_path)
        if os.path.exists(full):
            return full
    return os.path.join(video_base, video_path)


def load_video_frames(video_path, num_frames=32):
    from decord import VideoReader, cpu
    from PIL import Image
    import numpy as np

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def run_inference(model, tokenizer, image_processor, sample, video_base, num_frames=32):
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    video_path = find_video(sample["video_path"], video_base)
    frames = load_video_frames(video_path, num_frames=num_frames)
    question = sample["qa"][0]["question"] + POST_PROMPT

    image_tokens = (DEFAULT_IMAGE_TOKEN + "\n") * len(frames)
    prompt = image_tokens + question

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    images = process_images(frames, image_processor, model.config)
    if isinstance(images, list):
        images = [img.cuda().half() for img in images]
    else:
        images = images.cuda().half()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=[frame.size for frame in frames],
            do_sample=False,
            temperature=0,
            max_new_tokens=16,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_prediction(prediction, ground_truth):
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
    parser.add_argument("--video_base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)

    # PruneVid params
    parser.add_argument("--selected_layer", type=int, default=10,
                        help="LLM layer whose attention drives query-aware pruning")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Pruning intensity")
    parser.add_argument("--tau", type=float, default=0.8,
                        help="Attention threshold for relevant token selection")
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25,
                        help="Temporal segment fraction for stage-1 merging")
    parser.add_argument("--cluster_ratio", type=float, default=0.5,
                        help="Spatial cluster ratio for stage-1 merging")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model with PruneVid patches...")
    model, tokenizer, image_processor = load_model(args)

    print("Loading MotionBench...")
    samples = load_motionbench(args.meta_path)
    if args.limit:
        samples = samples[: args.limit]
    print(f"  {len(samples)} samples")

    results = []
    scores = []

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            prediction = run_inference(
                model, tokenizer, image_processor,
                sample, args.video_base, args.num_frames,
            )
        except Exception as e:
            print(f"  [WARN] sample {i} failed: {e}", file=sys.stderr)
            prediction = ""

        gt = sample["qa"][0]["answer"]
        score = score_prediction(prediction, gt)

        entry = {
            "doc_id": i,
            "category": sample.get("question_type", "Unknown"),
            "video_path": sample["video_path"],
            "ground_truth": gt,
            "prediction": prediction,
            "correct": score,
        }
        results.append(entry)
        if score is not None:
            scores.append(score)

    out_file = os.path.join(args.output_dir, "results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(scores)
    correct = sum(scores)
    accuracy = correct / total if total > 0 else 0.0
    na_count = len(results) - total

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total_scoreable": total,
        "total_na_skipped": na_count,
        "total_samples": len(results),
        "prunevid_params": {
            "selected_layer": args.selected_layer,
            "alpha": args.alpha,
            "tau": args.tau,
            "temporal_segment_ratio": args.temporal_segment_ratio,
            "cluster_ratio": args.cluster_ratio,
        },
    }
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)")

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results: {out_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ===========================================================================
# SAMPLED VARIANT — verbatim copy of stage1-llava-ov/videoitg-motionbenc/eval_videoitg_infer.py, with ONE behavioural change:
# generation is stochastic instead of greedy.
#
# The original under stage1-llava-ov/ is UNTOUCHED and still produces the gated
# greedy results that master-results.md depends on.
#
# These numbers are NOT comparable to the gated results and CANNOT pass the
# divergence gate — two sampled runs differ by chance, so "differs from
# baseline" no longer proves a method engaged. They exist to show the A/B/C/D
# answer DISTRIBUTION. Never merge them into master-results.md.
# ===========================================================================
"""
VideoITG Stage 2 × MotionBench — LLaVA-OV-7B inference with selected frames.

Reads frame_scores.jsonl produced by eval_videoitg_grounding.py (Stage 1).
Loads the top-K frame indices selected by VideoITG-8B and passes them to
LLaVA-OV-7B for question answering.  Falls back to uniform sampling when a
video has no grounding output.

Requires: dycoke11 conda env (standard LLaVA-OV setup)

Usage:
    python eval_videoitg_infer.py \\
        --model_path     /project/rhu/dpalfaro/weights/llava-ov-7b \\
        --meta_path      /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        --grounding_jsonl /project/rhu/dpalfaro/results/videoitg_grounding/frame_scores.jsonl \\
        --output_dir     /project/rhu/dpalfaro/results/videoitg_run1 \\
        [--num_frames 32] [--limit 50]
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm

# Sampling config; overwritten from CLI in main().
_SAMP = {"temperature": 0.7, "top_p": 0.9, "greedy": False}


# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Model loading — standard LLaVA-OV-7B
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Video loading — specific frame indices, subprocess-isolated
# ---------------------------------------------------------------------------
def load_frames_at_indices(video_path: str, frame_indices: list, fallback_n: int):
    """Load specific frame indices. Falls back to uniform if indices is empty."""
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, indices, fallback, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0))
            total = len(vr)
            idx = sorted(set(i for i in indices if 0 <= i < total)) if indices else []
            if not idx:
                idx = np.linspace(0, total - 1, fallback, dtype=int).tolist()
            frames = vr.get_batch(idx).asnumpy()
            q.put(("ok", frames))
        except Exception as e:
            q.put(("error", str(e)))

    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, frame_indices, fallback_n, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill()
        proc.join(timeout=5)
        raise RuntimeError(f"timeout (NFS stale?): {video_path}")
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == "error":
        raise RuntimeError(f"decode error: {video_path} — {data}")

    from PIL import Image
    return [Image.fromarray(f) for f in data]


# ---------------------------------------------------------------------------
# Inference — standard LLaVA-OV generate()
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames, question):
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle

    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    images = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    images = images.to(dtype=model.dtype, device="cuda")

    w, h = frames[0].size  # PIL: (width, height)
    image_sizes = [(h, w)] * len(frames)

    output_ids = model.generate(
        input_ids,
        images=[images],
        image_sizes=image_sizes,
        modalities=["video"],
        do_sample=(not _SAMP['greedy']),
        temperature=_SAMP['temperature'], top_p=_SAMP['top_p'],
        max_new_tokens=32,
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
    parser.add_argument("--grounding_jsonl", required=True,
                        help="frame_scores.jsonl from Stage 1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Max frames to pass to LLaVA-OV")
    parser.add_argument("--limit", type=int, default=None)
    # --- sampled-variant flags (not present in the original) ---------------
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true",
                        help="force greedy, for A/B against the gated run")
    args = parser.parse_args()
    _SAMP.update(temperature=args.temperature, top_p=args.top_p,
                 greedy=args.greedy)
    import random as _rnd
    import numpy as _np
    _rnd.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    grounding = {}
    with open(args.grounding_jsonl) as f:
        for line in f:
            rec = json.loads(line.strip())
            grounding[rec["idx"]] = rec.get("frame_indices", [])

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
    scores  = []

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        video_path   = find_video(sample["video_path"])
        question     = sample["qa"][0]["question"]
        ground_truth = sample["qa"][0]["answer"]
        q_type       = sample.get("question_type", "Unknown")
        frame_indices = grounding.get(i, [])[:args.num_frames]

        if video_path is None:
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            prediction = ""
        else:
            try:
                frames = load_frames_at_indices(
                    video_path, frame_indices, args.num_frames
                )
                prediction = run_inference(
                    tokenizer, model, image_processor, frames, question
                )
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}",
                      file=sys.stderr)
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
        "grounding_jsonl":  args.grounding_jsonl,
        # `enabled` lets scripts/check_run.py confirm the method actually ran.
        # VideoITG is two-stage: the grounding pass selects frames, this pass
        # consumes them. Engagement == a grounding file was supplied and used.
        "videoitg_params": {
            "enabled":       bool(args.grounding_jsonl),
            "stage":         "infer (consumes grounding frame_scores.jsonl)",
            "grounding_src": args.grounding_jsonl,
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

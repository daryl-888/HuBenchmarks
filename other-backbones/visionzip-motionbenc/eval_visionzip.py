#!/usr/bin/env python3
"""
VisionZip × MotionBench — CLIP-layer visual token reduction.

VisionZip (dvlab-research/VisionZip) selects two sets of tokens inside the
vision encoder before the LLM ever sees them:
  - Dominant tokens: highest CLS-attention score patches (most salient)
  - Contextual tokens: neighbouring patches preserving spatial context

The public API is a two-line monkey-patch:
    from visionzip import visionzip
    model = visionzip(model, dominant=54, contextual=10)

BACKBONE NOTE: VisionZip patches CLIPVisionTower. LLaVA-OV-7B uses SigLipVisionTower,
not CLIP. Two options for running:
  A) LLaVA-1.5-7B (CLIPVisionTower) — VisionZip's native target. Needs separate weights:
       huggingface-cli download liuhaotian/llava-v1.5-7b \
           --local-dir /project/rhu/dpalfaro/weights/llava-v1.5-7b
     Pass --model_path .../llava-v1.5-7b --conv_template llava_v1 --model_name llava_v1.5_7b
  B) LLaVA-OV-7B — try visionzip() and check if it silently no-ops or errors on SigLIP.
     The VisionZip repo added Qwen2.5-VL support (2025-05) suggesting some SigLIP handling
     may exist, but this is unverified for LLaVA-OV specifically.

Default here targets LLaVA-1.5-7B (option A) for correctness.

Source:  https://github.com/dvlab-research/VisionZip  (pip install visionzip)
Conda:   visionzip  (clone dycoke11, then pip install visionzip)
Weights: /project/rhu/dpalfaro/weights/llava-v1.5-7b  (download separately)
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
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, model_name: str, conv_template: str,
               dominant: int, contextual: int):
    from llava.model.builder import load_pretrained_model
    from visionzip import visionzip as apply_visionzip

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name,
        attn_implementation="sdpa",
    )
    model = apply_visionzip(model, dominant=dominant, contextual=contextual)
    model = model.cuda()
    model.eval()
    return tokenizer, model, image_processor, conv_template


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
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, conv_template, frames, question):
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

    # LLaVA-1.5 (liuhaotian/LLaVA): images is a plain tensor, no modalities/image_sizes kwargs.
    # For multiple frames, stack into [N, C, H, W] — the model treats each as a separate image
    # and the single <image> token in the prompt receives the concatenated visual features.
    images = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    images = images.to(dtype=model.dtype, device="cuda")

    output_ids = model.generate(
        input_ids,
        images=images,
        do_sample=False,
        temperature=0,
        max_new_tokens=16,
        use_cache=True,
    )

    # NOTE: LLaVA-1.5's generate() returns ONLY the newly generated tokens — the
    # prompt is not prepended (the <image> placeholder is expanded into visual
    # embeddings internally, so input_ids.shape[1] does not correspond to the
    # output's prefix length at all). Slicing by input_ids.shape[1] therefore
    # discarded the entire response and produced 100% EMPTY predictions.
    # Decode output_ids directly, exactly as the working DyCoke/FastV evals do.
    # Guard anyway in case a future version starts echoing the prompt.
    if output_ids.shape[1] > input_ids.shape[1] and \
            torch.equal(output_ids[:, :input_ids.shape[1]], input_ids):
        output_ids = output_ids[:, input_ids.shape[1]:]
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
    parser.add_argument("--model_name", default="llava_v1.5_7b",
                        help="Model name for load_pretrained_model (e.g. llava_v1.5_7b or llava_qwen)")
    parser.add_argument("--conv_template", default="llava_v1",
                        help="Conversation template (llava_v1 for LLaVA-1.5, qwen_1_5 for LLaVA-OV)")
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dominant", type=int, default=54,
                        help="Number of dominant CLIP tokens to keep (VisionZip default: 191; compressed: 54)")
    parser.add_argument("--contextual", type=int, default=10,
                        help="Number of contextual CLIP tokens to keep (VisionZip default: 30; compressed: 10)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor, conv_template = load_model(
        args.model_path,
        model_name=args.model_name,
        conv_template=args.conv_template,
        dominant=args.dominant,
        contextual=args.contextual,
    )

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
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            prediction = ""
        else:
            try:
                frames     = load_frames(video_path, args.num_frames)
                prediction = run_inference(
                    tokenizer, model, image_processor, conv_template, frames, question
                )
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}", file=sys.stderr)
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
        "visionzip_params": {
            "enabled": True,
            "dominant":    args.dominant,
            "contextual":  args.contextual,
            "num_frames":  args.num_frames,
        },
    }
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)", flush=True)

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:  {out_file}", flush=True)
    print(f"Summary:  {summary_file}", flush=True)


if __name__ == "__main__":
    main()

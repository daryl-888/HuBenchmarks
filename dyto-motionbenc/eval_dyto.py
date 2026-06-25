#!/usr/bin/env python3
"""
DyTo × MotionBench — native DyTo pipeline (ICCV 2025).

DyTo (Dynamic Token Merging) is a training-free zero-shot video understanding
method that optimizes token efficiency through:
  1. Hierarchical frame selection (temporal clustering)
  2. Bipartite token merging inside the LLM

Backbone: LLaVA-NeXT-Vicuna-7B (NOT LLaVA-OV-Qwen — different architecture)
  Weights: /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b
  Conv template: vicuna_v1

Source:     /project/rhu/dpalfaro/code/DYTO
PYTHONPATH: DYTO/dyto/llava (patched LLaVA) : DYTO (dataset, prompt, utils)
Conda env:  dyto

Env vars:
    TEMPORAL_AGGREGATION=cluster   DyTo token merging mode (default: cluster).
                                   Set to empty string to disable (baseline).

Setup on Carya:
    1. Clone DYTO:
       cd /project/rhu/dpalfaro/code && git clone https://github.com/Jam1ezhang/DYTO

    2. Create conda env:
       conda create --name dyto --clone dycoke11
       /project/rhu/dpalfaro/conda/envs/dyto/bin/pip install -e /project/rhu/dpalfaro/code/DYTO

    3. Download weights:
       git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b \
           /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b

Usage:
    TEMPORAL_AGGREGATION=cluster \\
    python eval_dyto.py \\
        --model-path /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b \\
        --meta-file  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        --output-dir /project/rhu/dpalfaro/results/dyto_run1 \\
        [--num-frames 32] [--limit 10]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Video loading — subprocess-isolated for NFS stale-handle safety
# ---------------------------------------------------------------------------
def load_video_frames(video_path: Path, num_frames: int) -> np.ndarray:
    """
    Load video frames in a subprocess with a 60s timeout.
    Returns (N, H, W, 3) uint8 numpy array.
    On NFS stale-handle timeout or decode error, returns black frames.
    """
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, q):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(p), ctx=cpu(0))
            total = len(vr)
            idx = np.linspace(0, total - 1, n, dtype=np.int64).tolist()
            frames = vr.get_batch(idx).asnumpy()
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
        LOGGER.warning("load_video_frames: timeout (NFS stale?), returning black frames: %s", video_path)
        return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == "error":
        LOGGER.warning("load_video_frames: decode error, returning black frames: %s — %s", video_path, data)
        return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
    return data


# ---------------------------------------------------------------------------
# Model loading — DyTo's patched LLaVA-NeXT (Vicuna backbone)
# ---------------------------------------------------------------------------
def load_model(model_path: str, model_base: str = None):
    """
    Load DyTo's patched LLaVA-NeXT model.
    Requires PYTHONPATH to include DYTO/dyto/llava (for `from llava...` imports).
    """
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init

    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        device=torch.cuda.current_device(),
        device_map="cuda",
    )
    model = model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Inference — DyTo native generate() with temporal_aggregation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames: list,
                  question: str, temporal_aggregation: str = None) -> str:
    """
    Run DyTo inference on a set of video frames.
    Uses vicuna_v1 conv template (DyTo's backbone).
    If temporal_aggregation is set, DyTo's token merging is applied.
    """
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    post_prompt = "\nAnswer with the option's letter from the given choices directly."
    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + post_prompt
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # Process images
    image_tensor = process_images(frames, image_processor, model.config)
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

    # Get original frame sizes
    w, h = frames[0].size
    image_sizes = [(h, w)] * len(frames)

    # Build generation kwargs
    gen_kwargs = dict(
        do_sample=False,
        temperature=0,
        max_new_tokens=16,
        use_cache=True,
    )

    # Pass temporal_aggregation if set (DyTo's token merging hook)
    if temporal_aggregation:
        gen_kwargs["temporal_aggregation"] = temporal_aggregation

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        **gen_kwargs,
    )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"


def find_video(video_path: str):
    """Search for video in MotionBench subdirectories."""
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


def score_prediction(prediction: str, ground_truth: str):
    """Score a prediction against ground truth. Returns None for NA samples."""
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
    parser = argparse.ArgumentParser(description="DyTo × MotionBench evaluation")
    parser.add_argument("--model-path", required=True,
                        help="Path to LLaVA-NeXT-Vicuna-7B weights")
    parser.add_argument("--model-base", default=None,
                        help="Optional model base path")
    parser.add_argument("--meta-file", required=True,
                        help="MotionBench JSONL metadata file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for results")
    parser.add_argument("--num-frames", type=int, default=32,
                        help="Number of frames to sample (default: 32)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of samples for test runs (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already in results.jsonl")
    args = parser.parse_args()

    # DyTo temporal aggregation mode from env var
    temporal_aggregation = os.environ.get("TEMPORAL_AGGREGATION", None)
    if temporal_aggregation:
        LOGGER.info("DyTo token merging enabled: TEMPORAL_AGGREGATION=%s", temporal_aggregation)
    else:
        LOGGER.info("DyTo token merging disabled (baseline mode)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "results.jsonl")

    # Resume support
    done = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                r = json.loads(line)
                done.add(r.get("uid", str(r.get("idx", ""))))
        LOGGER.info("Resuming: %d samples already done", len(done))

    # Load model
    LOGGER.info("Loading model from %s ...", args.model_path)
    tokenizer, model, image_processor = load_model(args.model_path, args.model_base)

    # Load samples
    samples = []
    with open(args.meta_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    LOGGER.info("Evaluating %d samples (%d frames each)",
                len(samples), args.num_frames)

    results = []
    scores = []

    with open(out_file, "a") as fout:
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            uid = sample.get("uid", str(i))
            if uid in done:
                continue

            video_rel_path = sample["video_path"]
            question = sample["qa"][0]["question"]
            gt = sample["qa"][0]["answer"]
            q_type = sample.get("question_type", "Unknown")

            prediction = ""
            video_path = find_video(video_rel_path)
            if video_path is None:
                LOGGER.warning("Video not found: %s", video_rel_path)
            else:
                try:
                    frames_np = load_video_frames(Path(video_path), args.num_frames)
                    frames_pil = [Image.fromarray(f) for f in frames_np]

                    prediction = run_inference(
                        tokenizer, model, image_processor,
                        frames_pil, question,
                        temporal_aggregation=temporal_aggregation,
                    )
                except Exception as e:
                    LOGGER.warning("Sample %d (%s): %s", i, video_rel_path, e)
                    torch.cuda.empty_cache()

            s = score_prediction(prediction, gt)
            rec = {
                "uid": uid,
                "idx": i,
                "video_path": video_rel_path,
                "question_type": q_type,
                "ground_truth": gt,
                "prediction": prediction,
                "correct": s,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            results.append(rec)
            if s is not None:
                scores.append(s)

    total = len(scores)
    correct = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0

    LOGGER.info("Accuracy: %d/%d = %.4f  (%d NA skipped)", correct, total, accuracy, na_count)

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total_scoreable": total,
        "total_na_skipped": na_count,
        "total_samples": len(results),
        "model": args.model_path,
        "num_frames": args.num_frames,
        "temporal_aggregation": temporal_aggregation,
    }
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)", flush=True)
    LOGGER.info("Results: %s", out_file)
    LOGGER.info("Summary: %s", summary_file)


if __name__ == "__main__":
    main()
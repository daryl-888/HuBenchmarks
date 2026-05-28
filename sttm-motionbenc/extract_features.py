#!/usr/bin/env python3
"""
Step 1 of 2: Extract visual features from MotionBench videos using LLaVA-OV-7B's
vision encoder.

STTM separates visual encoding from LLM decoding. This script runs the vision
tower over every MotionBench video and saves:
  {output_dir}/features/{vid_id}.pt   — tensor from vision encoder
  {output_dir}/metadata/{vid_id}.pkl  — frame counts, timestamps, question, answer

The output format matches STTM's VidQA_Loader_Feature expected layout so that
eval_motionbench.py can use STTM's existing feature-loading code.

Run this ONCE. Then run eval_motionbench.py (Step 2) with any sa_* config.

Usage (from STTM repo root):
    python /path/to/sttm-motionbenc/extract_features.py \
        --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
        --meta_path /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
        --video_base /project/rhu/MotionBench_Data/MotionBench \
        --output_dir /project/rhu/dpalfaro/sttm_features/motionbench \
        --num_frames 32
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Video loading — matches STTM's process_video_with_decord pattern
# ---------------------------------------------------------------------------
def load_frames(video_path: str, num_frames: int):
    """Uniformly sample num_frames from video, return list of PIL Images.

    Uses subprocess isolation so NFS stale handles can't hang the job.
    Raises RuntimeError on timeout or decode failure — caller's try/except
    skips the sample (no .pt/.pkl saved), so it gets retried on next run.
    """
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
            timestamps = [float(vr.get_frame_timestamp(i)[0]) for i in indices]
            q.put(('ok', (frames, timestamps)))
        except Exception as e:
            q.put(('error', str(e)))

    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frames, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill()
        proc.join(timeout=5)   # D-state child ignores SIGKILL; don't wait forever
        raise RuntimeError(f'load_frames: timeout (NFS stale?): {video_path}')
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == 'error':
        raise RuntimeError(f'load_frames: decode error: {video_path} — {data}')

    from PIL import Image
    frames, timestamps = data
    return [Image.fromarray(f) for f in frames], timestamps


def find_video(video_path: str, video_base: str):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(video_base, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


# ---------------------------------------------------------------------------
# Model loading — vision tower only (no STTM patch needed here)
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    from llava.model.builder import load_pretrained_model

    try:
        import flash_attn  # noqa: F401
        kwargs = {}
    except ImportError:
        kwargs = {"attn_implementation": "eager"}

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen", device_map="auto", **kwargs
    )
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_features(model, image_processor, frames):
    """
    Run vision encoder only. Returns tensor of shape [1, N_tokens, D].
    Matches what STTM's video_feat_llavavideo.py saves in {vid}.pt.
    """
    from llava.mm_utils import process_images

    if not hasattr(model.config, 'max_batch_size'):
        model.config.max_batch_size = 32

    images = process_images(frames, image_processor, model.config)
    if isinstance(images, list):
        images = torch.stack(images).cuda().half()
    else:
        images = images.cuda().half()

    # Run through vision tower + mm_projector to get visual tokens
    with torch.no_grad():
        image_features = model.encode_images(images)  # [N_frames*patches, D]

    # Return as [1, N, D] — consistent with STTM's saved format
    return image_features.unsqueeze(0).cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--meta_path", required=True,
                        help="Path to video_info.meta.jsonl")
    parser.add_argument("--video_base", required=True,
                        help="Root containing self-collected/ and public-dataset/")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save .pt and .pkl files")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N samples (for testing)")
    args = parser.parse_args()

    feat_dir = os.path.join(args.output_dir, "features")
    meta_dir = os.path.join(args.output_dir, "metadata")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Load samples
    samples = []
    with open(args.meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]
    print(f"Extracting features for {len(samples)} MotionBench samples")

    print("Loading model...")
    tokenizer, model, image_processor = load_model(args.model_path)

    skipped = 0
    for i, sample in enumerate(tqdm(samples, desc="Extracting")):
        vid_id = f"motionbench_{i:05d}"
        feat_path = os.path.join(feat_dir, f"{vid_id}.pt")
        meta_path = os.path.join(meta_dir, f"{vid_id}.pkl")

        # Skip already extracted
        if os.path.exists(feat_path) and os.path.exists(meta_path):
            skipped += 1
            continue

        video_path = find_video(sample["video_path"], args.video_base)
        if video_path is None:
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            continue

        try:
            t0 = time.time()
            frames, timestamps = load_frames(video_path, args.num_frames)
            video_feats = extract_features(model, image_processor, frames)
            elapsed = time.time() - t0

            # Save features — format matching STTM's .pt files
            torch.save({"video_feats": video_feats}, feat_path)

            # Save metadata — format matching STTM's .pkl files
            meta = {
                "vid_id": vid_id,
                "original_path": sample["video_path"],
                "num_frames": args.num_frames,
                "timestamps": timestamps,
                "extract_time": elapsed,
                # QA fields used by eval_motionbench.py
                "question": sample["qa"][0]["question"],
                "answer": sample["qa"][0]["answer"],
                "question_type": sample.get("question_type", "Unknown"),
            }
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)

        except Exception as e:
            print(f"  [ERROR] sample {i} ({sample['video_path']}): {e}",
                  file=sys.stderr)
            continue

    total_extracted = len(samples) - skipped
    print(f"\nDone. Extracted: {total_extracted}  Already existed: {skipped}")
    print(f"Features: {feat_dir}")
    print(f"Metadata: {meta_dir}")


if __name__ == "__main__":
    main()

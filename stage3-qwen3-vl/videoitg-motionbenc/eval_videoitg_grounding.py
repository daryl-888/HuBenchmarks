#!/usr/bin/env python3
"""
VideoITG Stage 1 × MotionBench — frame grounding using VideoITG-8B.

Scores up to 512 uniformly sampled frames per video for relevance to the
question.  Outputs a JSONL with the top-K frame indices (chronological order)
per sample — consumed by eval_videoitg_infer.py (Stage 2).

VideoITG-8B is an Eagle-family model (SigLIP + Qwen2 + sigmoid frame-scoring
head).  It is separate from the downstream LLaVA-OV used in Stage 2.

Requires: videoitg conda env
PYTHONPATH: /project/rhu/dpalfaro/code/VideoITG (eagle package)

Usage:
    python eval_videoitg_grounding.py \\
        --model_path /project/rhu/dpalfaro/weights/videoitg-8b \\
        --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        --output_dir /project/rhu/dpalfaro/results/videoitg_grounding \\
        [--num_frames_sample 512] [--num_frames_select 32] [--target_fps 2] [--limit 50]
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm


# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")


# ---------------------------------------------------------------------------
# Model loading — VideoITG-8B (Eagle family, eagle package)
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    from eagle.model.builder import load_pretrained_model
    from eagle.mm_utils import get_model_name_from_path

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None,
        get_model_name_from_path(model_path),
        device_map="cuda:0",
    )
    model.half().eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Video loading — fps-aware sampling, subprocess-isolated
# ---------------------------------------------------------------------------
def load_frames_for_grounding(video_path: str, num_frm: int, target_fps: int):
    """Returns (frames_np [F,H,W,3], original_frame_indices [F])."""
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, fps, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0), num_threads=4)
            total = len(vr)
            orig_fps = vr.get_avg_fps()
            step = max(1, round(orig_fps / fps))
            idx = list(range(0, total, step))
            if len(idx) > n:
                scale = len(idx) / n
                idx = [idx[round((i + 1) * scale - 1)] for i in range(n)]
            frames = vr.get_batch(idx).asnumpy()
            q.put(("ok", (frames, idx)))
        except Exception as e:
            q.put(("error", str(e)))

    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frm, target_fps, q))
    proc.start()
    try:
        status, data = q.get(timeout=120)
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
    return data  # (frames_np, frame_indices)


# ---------------------------------------------------------------------------
# Frame scoring — VideoITG-8B sigmoid head
# ---------------------------------------------------------------------------
@torch.inference_mode()
def score_frames(tokenizer, model, image_processor,
                 frames_np, frame_indices, question, num_topk: int):
    """Run VideoITG-8B; return top-K original frame indices (ascending)."""
    from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from eagle.mm_utils import tokenizer_image_token

    video_tensor = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.half().cuda()

    prompt = DEFAULT_IMAGE_TOKEN + question + "\n"
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [input_ids], batch_first=True, padding_value=pad_id
    ).cuda()
    attention_mask = input_ids.ne(pad_id).cuda()

    response = model(input_ids, attention_mask=attention_mask, images=[video_tensor])
    logits = response.logits[0].sigmoid().view(-1)

    _, sorted_pos = torch.sort(logits, descending=True)
    selected = []
    for pos in sorted_pos.tolist():
        if pos < len(frame_indices):
            selected.append(frame_indices[pos])
        if len(selected) >= num_topk:
            break

    return sorted(selected)  # chronological order for Stage 2


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def find_video(video_path: str):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames_sample", type=int, default=512,
                        help="Frames to sample per video for scoring")
    parser.add_argument("--num_frames_select", type=int, default=32,
                        help="Top-K frames to keep for Stage 2")
    parser.add_argument("--target_fps", type=int, default=2,
                        help="Target FPS for frame sampling")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading VideoITG-8B...", flush=True)
    tokenizer, model, image_processor = load_model(args.model_path)

    samples = []
    with open(args.meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    print(f"Scoring {len(samples)} samples", flush=True)

    out_file = os.path.join(args.output_dir, "frame_scores.jsonl")
    with open(out_file, "w") as fout:
        for i, sample in enumerate(tqdm(samples, desc="Grounding")):
            video_path = find_video(sample["video_path"])

            if video_path is None:
                print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
                frame_indices = []
            else:
                try:
                    frames_np, frame_idx = load_frames_for_grounding(
                        video_path, args.num_frames_sample, args.target_fps
                    )
                    frame_indices = score_frames(
                        tokenizer, model, image_processor,
                        frames_np, frame_idx,
                        sample["qa"][0]["question"],
                        args.num_frames_select,
                    )
                except Exception as e:
                    print(f"  [WARN] sample {i} ({sample['video_path']}): {e}",
                          file=sys.stderr)
                    frame_indices = []
                    torch.cuda.empty_cache()

            fout.write(json.dumps({
                "idx":          i,
                "video_path":   sample["video_path"],
                "frame_indices": frame_indices,
            }) + "\n")
            fout.flush()

    print(f"Frame scores: {out_file}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MDP3 x MotionBench — training-free list-wise frame selection (ICCV 2025).

MDP3 selects frames using:
  - Query relevance   (SigLip text-image similarity)
  - List-wise diversity (Determinantal Point Process)
  - Temporal sequentiality (Dynamic Programming)

Source:     /project/rhu/dpalfaro/code/MDP3
PYTHONPATH: MDP3 (vlmeval) : LLaVA-NeXT (llava loader from HoliTom)
Conda env:  mdp3

Backbone: LLaVA-OV-7B (Qwen1.5) — same as DyCoke/HoliTom/VideoITG for comparability.
  Weights: /project/rhu/dpalfaro/weights/llava-ov-7b
  Conv template: qwen_1_5

  To use Qwen2 backbone instead (llava-ov-7b-qwen2), change:
    --model-path  /project/rhu/dpalfaro/weights/llava-ov-7b-qwen2
    --conv-template qwen_2

--- ONE-TIME SETUP ON CARYA (login node, has internet) ---

1. Clone MDP3:
   cd /project/rhu/dpalfaro/code
   git clone https://github.com/sunh-23/MDP3

2. Create conda env (clone from holitom for compatible transformers):
   conda create --name mdp3 --clone holitom

3. Install MDP3 into the env:
   /project/rhu/dpalfaro/conda/envs/mdp3/bin/pip install -e /project/rhu/dpalfaro/code/MDP3
   /project/rhu/dpalfaro/conda/envs/mdp3/bin/pip install torchvision pysubs2

4. Cache SigLip model (MDP3 frame selector needs it offline):
   /project/rhu/dpalfaro/conda/envs/mdp3/bin/python3 -c "
   from transformers import AutoModel, AutoProcessor
   AutoModel.from_pretrained('google/siglip-so400m-patch14-384')
   AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')
   "

Notes:
- TRANSFORMERS_OFFLINE=1 is set in the sbatch — SigLip must be cached before the GPU job.
- MDP3 frame selector device follows the loaded SigLip model (cpu or cuda).
- pool_frames: how many frames to extract from video before MDP3 selection.
- select_frames: how many MDP3 picks from the pool (default 8).
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
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, q):
        try:
            import numpy as np
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
# MDP3 frame selection
# ---------------------------------------------------------------------------
def select_frames_mdp3(frames_np: np.ndarray, question: str, num_select: int) -> list:
    """
    Run MDP3 frame selector on frames_np (N, H, W, C uint8).
    Returns list of PIL Images (length == num_select).

    MDP3.select() API (from vlmeval/smp/mdp3_frame_selector.py):
      selector = MDP3(num_frames=K)
      selected = selector.select(pil_frame_list, question_text)
      # returns list of PIL Images of length K

    If MDP3 API changes, verify against the cloned repo on Carya.
    """
    from vlmeval.smp.mdp3_frame_selector import MDP3

    pil_frames = [Image.fromarray(f) for f in frames_np]
    selector = MDP3(num_frames=num_select)
    selected = selector.select(pil_frames, question)

    # Normalize: if select() returns indices, convert to PIL
    if selected and isinstance(selected[0], int):
        selected = [pil_frames[i] for i in selected]

    # Fallback: if fewer frames returned than requested, pad with uniform sample
    if len(selected) < num_select:
        LOGGER.warning("MDP3 returned %d frames (expected %d), padding uniformly", len(selected), num_select)
        idxs = np.linspace(0, len(pil_frames) - 1, num_select, dtype=int)
        selected = [pil_frames[i] for i in idxs]

    return selected


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, conv_template: str):
    from llava.model.builder import load_pretrained_model

    model_name = "llava_qwen"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name,
        attn_implementation="sdpa",
    )
    model = model.cuda().eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames: list,
                  question: str, conv_template: str) -> str:
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    post_prompt = "\nAnswer with the option's letter from the given choices directly."
    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + post_prompt
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
def find_video(video_path: str, video_roots: list) -> str | None:
    for root in video_roots:
        for subdir in ("self-collected", "public-dataset", ""):
            full = os.path.join(root, subdir, video_path) if subdir else os.path.join(root, video_path)
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
    parser.add_argument("--meta-file",      required=True,
                        help="MotionBench JSONL metadata file")
    parser.add_argument("--video-roots",    nargs="+", required=True,
                        help="Root dir(s) containing self-collected/ and public-dataset/")
    parser.add_argument("--model-path",     required=True,
                        help="Path to LLaVA-OV weights")
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--conv-template",  default="qwen_1_5",
                        help="qwen_1_5 for llava-ov-7b (default), qwen_2 for llava-ov-7b-qwen2")
    parser.add_argument("--pool-frames",    type=int, default=32,
                        help="Frames extracted from video before MDP3 selection (default: 32)")
    parser.add_argument("--select-frames",  type=int, default=8,
                        help="Frames MDP3 selects from the pool (default: 8)")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Cap number of samples (for test runs)")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip samples already in predictions.jsonl")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "predictions.jsonl")

    done = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                r = json.loads(line)
                done.add(r.get("uid", str(r.get("idx", ""))))
        LOGGER.info("Resuming: %d samples already done", len(done))

    LOGGER.info("Loading model from %s ...", args.model_path)
    tokenizer, model, image_processor = load_model(args.model_path, args.conv_template)

    samples = []
    with open(args.meta_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    LOGGER.info("Evaluating %d samples (pool=%d select=%d)",
                len(samples), args.pool_frames, args.select_frames)

    results = []
    scores  = []

    with open(out_file, "a") as fout:
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            uid = sample.get("uid", str(i))
            if uid in done:
                continue

            video_path = find_video(sample["video_path"], args.video_roots)
            question   = sample["qa"][0]["question"]
            gt         = sample["qa"][0]["answer"]
            q_type     = sample.get("question_type", "Unknown")

            prediction = ""
            if video_path is None:
                LOGGER.warning("Video not found: %s", sample["video_path"])
            else:
                try:
                    frames_np  = load_video_frames(Path(video_path), args.pool_frames)
                    frames     = select_frames_mdp3(frames_np, question, args.select_frames)
                    prediction = run_inference(
                        tokenizer, model, image_processor, frames, question, args.conv_template
                    )
                except Exception as e:
                    LOGGER.warning("Sample %d (%s): %s", i, sample["video_path"], e)
                    torch.cuda.empty_cache()

            s = score_prediction(prediction, gt)
            rec = {
                "uid":           uid,
                "idx":           i,
                "video_path":    sample["video_path"],
                "question_type": q_type,
                "ground_truth":  gt,
                "prediction":    prediction,
                "correct":       s,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            results.append(rec)
            if s is not None:
                scores.append(s)

    total    = len(scores)
    correct  = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0

    LOGGER.info("Accuracy: %d/%d = %.4f  (%d NA skipped)", correct, total, accuracy, na_count)

    summary = {
        "accuracy":         accuracy,
        "correct":          correct,
        "total_scoreable":  total,
        "total_na_skipped": na_count,
        "total_samples":    len(results),
        "model":            args.model_path,
        "conv_template":    args.conv_template,
        "mdp3_params": {
            "pool_frames":   args.pool_frames,
            "select_frames": args.select_frames,
        },
    }
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Results: %s", out_file)
    LOGGER.info("Summary: %s", summary_file)


if __name__ == "__main__":
    main()

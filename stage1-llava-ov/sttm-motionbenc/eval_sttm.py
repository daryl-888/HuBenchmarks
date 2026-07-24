#!/usr/bin/env python3
"""
STTM × MotionBench — native STTM pipeline.

Applies STTM's replace_qwen2_with_quadtree_attn() before loading the model,
then iterates MotionBench directly.  Uses STTM's own load_pretrained_model
(torch_dtype="bfloat16", attn_implementation="flash_attention_2") and the
native generate() signature (prompt_stat, image_sizes, modalities).

Requires: sttm_new conda env (Python 3.10, torch 2.5.1+cu121, flash-attn 2.7.3)
PYTHONPATH: /project/rhu/dpalfaro/code/STTM only — llava 1.7.0.dev0 is
installed into sttm_new, so DyCoke is NOT needed on the path.

Usage:
    PYTHONPATH=/project/rhu/dpalfaro/code/STTM \\
    python eval_sttm.py \\
        --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \\
        --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        --output_dir /project/rhu/dpalfaro/results/sttm_v2_run1 \\
        [--num_frames 32] [--limit 50] \\
        [--sa_start_layer_idx 2] [--sa_tree_thresh 0.85] \\
        [--sa_tree_temporal_thresh 0.65] [--sa_tree_root_level 1]
"""

import argparse
import json
import os
import re
import sys

import math
import torch
from tqdm import tqdm


# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# STTM quadtree patch — must be applied before model load
# ---------------------------------------------------------------------------
def apply_sttm_patch(args):
    from token_merging_monkey_patch.quadtree_attn_monkey_patch import (
        replace_qwen2_with_quadtree_attn,
    )
    replace_qwen2_with_quadtree_attn(
        sa_start_layer_idx=args.sa_start_layer_idx,
        sa_tree_thresh=args.sa_tree_thresh,
        sa_tree_temporal_thresh=args.sa_tree_temporal_thresh,
        sa_tree_root_level=args.sa_tree_root_level,
    )
    print(
        f"STTM patch applied: layer={args.sa_start_layer_idx} "
        f"thresh={args.sa_tree_thresh} temporal={args.sa_tree_temporal_thresh} "
        f"root_level={args.sa_tree_root_level}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Model loading — STTM native (bfloat16 + flash_attention_2)
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.config.max_batch_size = 256
    # Guard: ensure bfloat16 regardless of which builder version handled the kwarg
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
    # LLaVA-OV-7B config attributes STTM's forward pass reads
    if not hasattr(model.config, "mm_spatial_pool_stride"):
        model.config.mm_spatial_pool_stride = 2
    if not hasattr(model.config, "mm_newline_position"):
        model.config.mm_newline_position = "no_token"
    model.eval()
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
# Inference — STTM native generate() with prompt_stat
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

    # Locate image token to compute prompt_stat boundaries
    img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if len(img_pos) == 0:
        raise RuntimeError("No image token found in input_ids")
    sys_len  = img_pos[0].item()
    inst_len = input_ids.shape[1] - sys_len - 1  # exclude the single image placeholder

    # Process frames: use image_processor directly (no LLaVA-OV dynamic tiling).
    # process_images() from llava.mm_utils tiles each frame into N sub-images,
    # producing [T, N, C, H, W] which STTM's conv2d patch_embed cannot accept.
    # Direct preprocess() gives [T, C, H, W] — one tensor per frame.
    images = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    images = images.to(torch.bfloat16).cuda()
    n_frames = images.shape[0]

    # How many visual tokens per frame land in the LLM after spatial pooling.
    # SigLIP-SO400M-patch14-384: 384/14=27 patches/side. pool_stride=2 → 13/side.
    pool_stride = getattr(model.config, "mm_spatial_pool_stride", 2)
    try:
        vt_cfg = model.model.vision_tower.vision_tower.config
        patch_per_side = vt_cfg.image_size // vt_cfg.patch_size  # 27
    except AttributeError:
        patch_per_side = 27
    spatial_per_side = math.ceil(patch_per_side / pool_stride)  # ceil(27/2)=14
    tokens_per_frame_llm = spatial_per_side * spatial_per_side  # 169

    # STTM reads boundaries from model.model attrs AND from prompt_stat["video"].
    # llava_qwen.py:generate() does: self.model.image_token_length = prompt_stat["video"]
    # "video" must be the visual token count excluding any trailing newline so that
    # it divides exactly as T * H * W = n_frames * 14 * 14 = 3136.
    total_visual_tokens = n_frames * tokens_per_frame_llm
    model.model.image_token_start_index = torch.tensor(sys_len, dtype=torch.long)
    model.model.image_token_length      = torch.tensor(total_visual_tokens, dtype=torch.long)
    model.model.num_frame               = torch.tensor(n_frames, dtype=torch.long)
    # STTM's generate() always recomputes: video = inputs_embeds.size(1) - sys - inst
    # LLaVA-OV appends 1 grid-newline token at the end of the visual block, making
    # the visual block 3137 = 16*196+1. Adding +1 to inst absorbs that newline so
    # STTM sees video = 3137-1 = 3136 = T*H*W = 16*14*14 (exactly divisible).
    prompt_stat = {
        "sys":   sys_len,
        "inst":  inst_len + 1,
        "frame": n_frames,
    }

    h, w = images.shape[-2], images.shape[-1]
    image_sizes = [(h, w)] * n_frames

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    try:
        from llava.mm_utils import KeywordsStoppingCriteria
        stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)]
    except (ImportError, AttributeError):
        stopping_criteria = []

    output_ids, _ = model.generate(
        inputs=input_ids,
        images=[images],
        image_sizes=image_sizes,
        modalities=["video"],
        stopping_criteria=stopping_criteria,
        prompt_stat=prompt_stat,
        max_new_tokens=32,
        num_beams=1,
        do_sample=False,
        use_cache=True,
    )

    # STTM's generate() uses inputs_embeds internally, so HF returns only
    # the newly generated tokens (not prompt+output). Decode directly.
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
    parser.add_argument("--meta_path", required=True,
                        help="Path to video_info.meta.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N samples (smoke test)")

    # STTM hyperparameters — defaults from paper for LLaVA-OV-7B
    parser.add_argument("--sa_start_layer_idx",      type=int,   default=2)
    parser.add_argument("--sa_tree_thresh",           type=float, default=0.85)
    parser.add_argument("--sa_tree_temporal_thresh",  type=float, default=0.65)
    parser.add_argument("--sa_tree_root_level",       type=int,   default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    apply_sttm_patch(args)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(args.model_path)

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
        "accuracy":           accuracy,
        "correct":            correct,
        "total_scoreable":    total,
        "total_na_skipped":   na_count,
        "total_samples":      len(results),
        "sttm_params": {
            "enabled": True,
            "sa_start_layer_idx":      args.sa_start_layer_idx,
            "sa_tree_thresh":          args.sa_tree_thresh,
            "sa_tree_temporal_thresh": args.sa_tree_temporal_thresh,
            "sa_tree_root_level":      args.sa_tree_root_level,
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

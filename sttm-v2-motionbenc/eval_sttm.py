#!/usr/bin/env python3
"""
STTM × MotionBench — thin wrapper, no lmms_eval.

Calls STTM's unmodified replace_qwen2_with_quadtree_attn() on LLaVA-OV-7B
(STTM's native base model), then iterates MotionBench directly.
This file provides only: STTM patch application, dataset iteration,
subprocess video loading, and NA-skip scoring.

Three compatibility shims bridge STTM (written for transformers 4.40.0.dev0)
with the installed env (4.45.2). None of them modify STTM's code:
  1. model.config.max_batch_size = 32    (LlavaQwenConfig lacks this attr)
  2. pre-forward hook on model.model     (truncates image_token_length to T×H×W)
  3. prompt_stat built from token counts (STTM's generate() requires this dict)

After the STTM monkey-patch is imported, STTM's root is removed from sys.path
to prevent its partial llava/ directory from shadowing the installed package.

Usage:
    PYTHONPATH=/project/rhu/dpalfaro/code/STTM:$PYTHONPATH \\
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
from pathlib import Path

import torch
from tqdm import tqdm


VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# STTM quadtree patch
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

    # STTM's repo ships a partial llava/ package (mm_utils.py, constants.py,
    # conversation.py) that shadows the env's installed llava when STTM is on
    # PYTHONPATH.  Now that we've imported what we need from STTM, remove it
    # so that subsequent `from llava.mm_utils import process_images` etc. use
    # the complete, installed package instead.
    import sys
    sttm_roots = [
        p for p in sys.path
        if p.rstrip("/").rstrip("\\").endswith("STTM")
    ]
    for p in sttm_roots:
        sys.path.remove(p)
    stale_llava = [k for k in list(sys.modules) if k == "llava" or k.startswith("llava.")]
    for k in stale_llava:
        del sys.modules[k]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    from llava.model.builder import load_pretrained_model

    # load_pretrained_model defaults to attn_implementation="flash_attention_2"
    # which requires the flash_attn package (not installed in the sttm env).
    # Pass "sdpa" instead: PyTorch's scaled_dot_product_attention is
    # memory-efficient (O(n) like flash attention) and needs no extra package.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen", device_map="auto",
        attn_implementation="sdpa",
    )

    # LlavaQwenConfig lacks max_batch_size; STTM's Qwen2Model_forward reads it.
    if not hasattr(model.config, "max_batch_size"):
        model.config.max_batch_size = 32

    # Pre-forward hook: correct image_token_length to be exactly T*H*W.
    # LLaVA-OV's prepare_inputs_labels_for_multimodal adds an image_newline
    # token, making the visual token count T*H*W+1. STTM's einops rearrange
    # requires exactly T*H*W. This hook truncates at the point of use.
    def _fix_image_token_length(module, _args):
        if (
            hasattr(module, "image_token_length")
            and hasattr(module, "num_frame")
            and isinstance(module.image_token_length, torch.Tensor)
            and isinstance(module.num_frame, torch.Tensor)
        ):
            img_len = module.image_token_length.item()
            n_frames = module.num_frame.item()
            if n_frames > 0 and img_len % n_frames != 0:
                module.image_token_length = torch.tensor(
                    (img_len // n_frames) * n_frames,
                    dtype=torch.long,
                )

    model.model.register_forward_pre_hook(_fix_image_token_length)

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
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames, question):
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    # Build the conversation prompt
    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    # Tokenise — IMAGE_TOKEN_INDEX is still a single placeholder here
    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if len(img_pos) == 0:
        raise RuntimeError("No image token found in input_ids")
    sys_tokens = img_pos[0].item()

    # Process video frames
    images = process_images(frames, image_processor, model.config)
    if isinstance(images, list):
        images = torch.stack(images)
    images = images.cuda().half()
    n_frames = images.shape[0]

    # Set the three attributes STTM's patched Qwen2Model_forward reads from
    # self (model.model) to locate image tokens and build the quadtree.
    # image_token_length must be exactly T*H*W = T*729 (SigLIP-SO400M 384px,
    # patch 14 → 27*27=729 per frame). LLaVA-OV adds an image_newline token
    # per frame (total 730/frame), but STTM's einops rearrange needs 729.
    # We set 729 explicitly; the pre-forward hook is kept as a fallback.
    try:
        vt_cfg = model.model.vision_tower.vision_tower.config
        tokens_per_frame = (vt_cfg.image_size // vt_cfg.patch_size) ** 2  # 27^2=729
    except AttributeError:
        tokens_per_frame = 729  # SigLIP-SO400M-patch14-384 fallback
    model.model.image_token_start_index = torch.tensor(sys_tokens,                  dtype=torch.long)
    model.model.image_token_length      = torch.tensor(n_frames * tokens_per_frame, dtype=torch.long)
    model.model.num_frame               = torch.tensor(n_frames,                    dtype=torch.long)

    input_len = input_ids.shape[1]

    result = model.generate(
        inputs=input_ids,
        images=images,
        do_sample=False,
        temperature=0,
        max_new_tokens=16,
        use_cache=True,
    )
    sequences = result[0] if isinstance(result, (tuple, list)) else result

    # Decode only the newly generated tokens
    generated = sequences[:, input_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


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

    # Must apply patch BEFORE loading the model
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

    # Save per-sample results
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
            "sa_start_layer_idx":     args.sa_start_layer_idx,
            "sa_tree_thresh":         args.sa_tree_thresh,
            "sa_tree_temporal_thresh": args.sa_tree_temporal_thresh,
            "sa_tree_root_level":     args.sa_tree_root_level,
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

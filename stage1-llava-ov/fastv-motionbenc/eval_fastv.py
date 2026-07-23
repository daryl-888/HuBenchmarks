#!/usr/bin/env python3
"""
FastV × MotionBench — attention-based visual token pruning inside the LLM.

FastV (arXiv 2403.06764) prunes image tokens at LLM layer K based on the
average attention they receive from all other tokens. Tokens in the bottom R%
of received-attention scores are dropped; computation continues with the reduced
set through layers K+1 onward. No retraining required.

PORTING STATUS: FastV's reference implementation patches modeling_llama.py in a
custom transformers fork. LLaVA-OV uses Qwen1.5 (modeling_qwen2.py), not LLaMA.
The porting work is isolated to apply_fastv() below — everything else is ready.

Port reference: FastV/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py
Target:         Qwen2Model.forward() in transformers/models/qwen2/modeling_qwen2.py
Algorithm (identical for both):
  1. Run the first K transformer layers normally.
  2. After layer K, compute for each token the average attention it received:
       attn_score[i] = mean over heads of: sum_j attn_weight[j, i]  (column sum)
  3. Among image tokens only (indices image_start : image_start + image_len),
     keep those with attn_score in the top (1 - fastv_r) fraction.
  4. Remove the pruned tokens from the hidden state and all subsequent KV caches.
  5. Continue forward through layers K+1 ... L with the shortened sequence.

Source:  https://github.com/chenllliang/FastV
Conda:   fastv  (clone dycoke11, then install FastV's custom transformers fork)
Weights: /project/rhu/dpalfaro/weights/llava-ov-7b  (LLaVA-OV-7B, Qwen 1.5)
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm


VIDEO_BASE  = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# FastV — attention-based visual-token pruning on LLaVA-OV's Qwen2 backbone
# ---------------------------------------------------------------------------
#
# Ported to match the token-location convention already used by this LLaVA build
# (see DyCoke's modeling_qwen2.py): image tokens occupy the contiguous span
#   [image_token_start_index : image_token_start_index + image_token_length]
# where the prefix length is 14 (qwen conv-template system+user preamble) and the
# image length is (total prefill tokens) - (text tokens). During the prefill
# forward we rank those image tokens by the attention they receive at layer K and
# drop the bottom fastv_r fraction for every layer > K by masking them out. This
# is the FastV paper's exact policy (one-shot, fixed layer K), implemented with an
# additive attention mask so no KV-cache surgery is needed — the pruned tokens
# simply contribute nothing from layer K+1 onward.

IMAGE_TOKEN_START_INDEX = 14  # qwen_1_5 preamble length before the image block


def apply_fastv(model, fastv_k: int, fastv_r: float):
    """
    Enable FastV pruning on this LLaVA-OV Qwen2 model.

    Implementation strategy: this LLaVA build already ships a patched
    modeling_qwen2.py whose Qwen2Model.forward supports in-loop KV-cache pruning
    (it is what DyCoke uses). We reuse that exact machinery with FastV's simpler,
    static policy — prune ONCE at layer `fastv_k`, keeping the top (1 - fastv_r)
    fraction of image tokens ranked by received attention — by installing a
    `fastv` config block that the patched forward acts on. The actual per-layer
    slicing lives in the modeling file (fastv_prune()), mirroring dycoke_pruning().

    fastv_k: LLM layer index at which to prune (0-indexed). Paper default: 2.
    fastv_r: fraction of image tokens to DROP (least-attended). Paper default: 0.5.

    Requires attn_implementation="eager" so per-layer attention weights exist,
    and the FastV-patched modeling_qwen2 on PYTHONPATH ahead of DyCoke's.
    """
    inner = model.model  # Qwen2Model (FastV-patched build)
    if not hasattr(inner, "enable_fastv"):
        raise RuntimeError(
            "This Qwen2Model has no enable_fastv() — the FastV-patched "
            "modeling_qwen2.py is not on PYTHONPATH ahead of DyCoke's. "
            "Check the sbatch PYTHONPATH order."
        )
    inner.enable_fastv(
        fastv_k=fastv_k,
        fastv_r=fastv_r,
        image_token_start_index=IMAGE_TOKEN_START_INDEX,
    )
    model.config._fastv_enabled = True
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, fastv_k: int, fastv_r: float, enable_fastv: bool):
    from llava.model.builder import load_pretrained_model

    # FastV requires eager attention so attn weights are accessible
    attn_impl = "eager" if enable_fastv else "sdpa"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        attn_implementation=attn_impl,
    )
    if enable_fastv:
        model = apply_fastv(model, fastv_k=fastv_k, fastv_r=fastv_r)
    model = model.cuda()
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
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

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
    parser.add_argument("--fastv", action="store_true",
                        help="Apply FastV pruning (requires Qwen port — see apply_fastv())")
    parser.add_argument("--fastv_k", type=int, default=2,
                        help="LLM layer at which to prune (FastV paper default: 2)")
    parser.add_argument("--fastv_r", type=float, default=0.5,
                        help="Fraction of image tokens to drop (FastV paper default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(
        args.model_path,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        enable_fastv=args.fastv,
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
                prediction = run_inference(tokenizer, model, image_processor, frames, question)
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
        "fastv_params": {
            "enabled":  args.fastv,
            "fastv_k":  args.fastv_k,
            "fastv_r":  args.fastv_r,
            "num_frames": args.num_frames,
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

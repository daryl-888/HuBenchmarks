#!/usr/bin/env python3
"""
VisionZip (CONTEXTUAL-ONLY) x MotionBench -- LLaVA-OV port.

VisionZip as published (dvlab-research/VisionZip) is two halves:
  1. DOMINANT tokens -- top-k by CLS-attention in the vision tower.
     NOT PORTABLE: LLaVA-OV's vision tower is SigLIP (google/siglip-so400m-
     patch14-384), which has no CLS token -- the signal this half is defined
     on does not exist. Same limitation as the Qwen3-VL port
     (stage3-qwen3-vl/visionzip-motionbenc/eval_visionzip_qwen3vl.py).
  2. CONTEXTUAL tokens -- merge remaining tokens by cosine similarity.
     This half is backbone-agnostic and is what this script implements.

Because the dominant half is absent, this MUST be reported as "VisionZip
(contextual-only)" -- a documented partial, never as "VisionZip" (mirrors
the existing convention for the Qwen3-VL cell in Table 2).

Hook point: LlavaMetaForCausalLM.get_2dPool(image_feature, stride) is the
function LLaVA-OV's video path uses to spatially pool each frame's raw
SigLIP patches (729/frame from a 27x27 grid) down before they enter the LLM
(llava_arch.py, encode_multimodals()). Unlike the Qwen3-VL port -- which had
to preserve sequence length because Qwen3VLModel.forward() does a
masked_scatter keyed to a fixed video-token-placeholder count -- LLaVA-OV
just concatenates whatever length of image_feature it's handed into the
token stream, so this port can genuinely SHRINK the token count instead of
merging-in-place at a fixed size. Retention ratio is applied to whatever
get_2dPool's own output size turns out to be at runtime (no need to
hardcode the pool stride LLaVA-OV actually uses for this checkpoint).

Source:  https://github.com/dvlab-research/VisionZip
Backbone: LLaVA-OV-7B-Qwen2 (llava-ov-7b-qwen2 weights), Conv template: qwen_2
"""
import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm

VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# VisionZip (contextual-only) -- hook LLaVA-OV's get_2dPool
# ---------------------------------------------------------------------------
def apply_visionzip_contextual(model, retention_ratio: float = 0.15):
    """
    Wrap model.get_2dPool (bound method, llava_arch.py's LlavaMetaForCausalLM
    mixin) so its output -- (num_frames, pooled_tokens_per_frame, hidden),
    the per-frame token stream LLaVA-OV's video path feeds to encode_
    multimodals() -- gets merged down to retention_ratio of its own size,
    per frame, by cosine similarity of the (post-pool) features themselves.
    No CLS-attention metric is available (SigLIP has no CLS token), so this
    uses the same features-as-basis fallback the Qwen3-VL port uses when its
    own metric can't be recovered -- a documented approximation, not the
    authors' exact "contextual" step (which merges by CLIP CLS-adjacent key
    vectors), but the closest backbone-agnostic analogue.
    """
    import types
    import torch.nn.functional as F

    orig_get_2dpool = model.get_2dPool
    state = {"logged": False}

    def merge_frame(feat: torch.Tensor, c_num: int) -> torch.Tensor:
        n = feat.shape[0]
        if c_num >= n:
            return feat
        basis = F.normalize(feat.float(), dim=-1)
        step = max(1, n // c_num)
        target_idx = torch.arange(0, n, step, device=feat.device)[:c_num]
        keep_mask = torch.ones(n, dtype=torch.bool, device=feat.device)
        keep_mask[target_idx] = False
        sim = basis[keep_mask] @ basis[target_idx].t()
        assign = sim.argmax(dim=1)
        merged = feat[target_idx].clone().float()
        counts = torch.ones(c_num, 1, device=feat.device)
        src = feat[keep_mask].float()
        merged.index_add_(0, assign, src)
        counts.index_add_(0, assign, torch.ones(src.shape[0], 1, device=feat.device))
        merged = (merged / counts).to(feat.dtype)
        return merged

    def patched_get_2dPool(self, image_feature, stride=2):
        pooled = orig_get_2dpool(image_feature, stride)
        num_frames, n, c = pooled.shape
        c_num = max(1, round(n * retention_ratio))
        if c_num >= n:
            return pooled
        out = torch.stack([merge_frame(pooled[f], c_num) for f in range(num_frames)], dim=0)
        if not state["logged"]:
            import logging
            logging.warning("VisionZip-contextual(LLaVA-OV) ACTIVE: get_2dPool per-frame "
                            "%d -> %d tokens (%.1f%% retention, %d frames)",
                            n, c_num, 100.0 * c_num / n, num_frames)
            state["logged"] = True
        return out

    model.get_2dPool = types.MethodType(patched_get_2dPool, model)
    return model


# ---------------------------------------------------------------------------
# Model loading (identical pattern to eval_fastv.py / eval_prunevid_ov.py)
# ---------------------------------------------------------------------------
def load_model(model_path: str, enable_visionzip: bool, retention_ratio: float):
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        attn_implementation="sdpa",
    )
    if enable_visionzip:
        model = apply_visionzip_contextual(model, retention_ratio=retention_ratio)
    model = model.cuda()
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Video loading -- subprocess-isolated for NFS stale-handle safety
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
# Inference (identical to eval_fastv.py, minus the FastV-specific text-tail hook)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames, question,
                  conv_template: str = "qwen_2"):
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
    parser.add_argument("--conv_template", default="qwen_2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--visionzip", action="store_true", help="apply VisionZip contextual-only merge")
    parser.add_argument("--retention_ratio", type=float, default=0.15)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(
        args.model_path, enable_visionzip=args.visionzip, retention_ratio=args.retention_ratio,
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
    scores = []
    per_category = {}
    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        video_path = find_video(sample["video_path"])
        question = sample["qa"][0]["question"]
        ground_truth = sample["qa"][0]["answer"]
        q_type = sample.get("question_type", "Unknown")
        if video_path is None:
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            prediction = ""
        else:
            try:
                frames = load_frames(video_path, args.num_frames)
                prediction = run_inference(tokenizer, model, image_processor, frames, question,
                                           conv_template=args.conv_template)
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}", file=sys.stderr)
                prediction = ""
                torch.cuda.empty_cache()
        s = score_prediction(prediction, ground_truth)
        results.append({
            "idx": i, "video_path": sample["video_path"], "question_type": q_type,
            "ground_truth": ground_truth, "prediction": prediction, "correct": s,
        })
        if s is not None:
            scores.append(s)
            per_category[q_type] = per_category.get(q_type, {"correct": 0, "total": 0})
            per_category[q_type]["total"] += 1
            per_category[q_type]["correct"] += s
    out_file = os.path.join(args.output_dir, "results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    total = len(scores)
    correct = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "accuracy": accuracy, "correct": correct, "total_scoreable": total,
        "total_na_skipped": na_count, "total_samples": len(results),
        "model": args.model_path, "num_frames": args.num_frames,
        "conv_template": args.conv_template,
        "visionzip_params": {"enabled": bool(args.visionzip), "variant": "contextual-only",
                             "retention_ratio": args.retention_ratio,
                             "note": "PARTIAL: dominant half needs a CLS token SigLIP lacks"},
        "per_category": per_category,
    }
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)", flush=True)
    for cat in sorted(per_category.keys()):
        c = per_category[cat]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0.0
        print(f"  {cat}: {c['correct']}/{c['total']} = {acc:.4f}", flush=True)
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:  {out_file}", flush=True)
    print(f"Summary:  {summary_file}", flush=True)


if __name__ == "__main__":
    main()

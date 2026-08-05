#!/usr/bin/env python3
"""
Qwen3-VL Baseline × MotionBench — ovqwen3.

Qwen3VLForConditionalGeneration is a native HuggingFace model, NOT a LLaVA fork.
Uses transformers AutoProcessor + from_pretrained (no llava.model.builder).

Backbone: Qwen/Qwen3-VL-8B-Instruct
Weights: /project/rhu/dpalfaro/weights/qwen3-vl-8b
Uses: AutoProcessor for video preprocessing, model.generate() for inference

This is a baseline eval — no model compression applied. All ovqwen3 models
start here, then compression methods are ported later.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Model loading — Qwen3-VL native HuggingFace
# ---------------------------------------------------------------------------

def _load_mdp3_class():
    """
    Load MDP3's frame selector WITHOUT importing the `vlmeval` package.

    `from vlmeval.smp.mdp3_frame_selector import MDP3` triggers
    vlmeval/__init__ -> smp/__init__ -> ... -> vlm/idefics.py, which does
        from transformers import AutoModelForVision2Seq
    That symbol was REMOVED in transformers 5.x — but Qwen3-VL *requires*
    transformers 5.x, so the package chain can never import in this env.

    The selector module itself only needs torch / PIL / transformers, so we load
    the single file directly by path and skip the package entirely.
    """
    import importlib.util
    import os
    import sys

    mdp3_root = os.environ.get("SRC_MDP3", "/project/rhu/dpalfaro/code/MDP3")
    path = os.path.join(mdp3_root, "vlmeval", "smp", "mdp3_frame_selector.py")
    if not os.path.exists(path):
        raise ImportError(f"MDP3 selector not found at {path} (set $SRC_MDP3)")
    spec = importlib.util.spec_from_file_location("_mdp3_frame_selector", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mdp3_frame_selector"] = mod
    spec.loader.exec_module(mod)
    return mod.MDP3


def select_frames_mdp3(pil_frames, question: str, num_select: int):
    """
    MDP3 (ICCV 2025) frame selection, ported to Qwen3-VL.

    MDP3 is the ONE method in this suite that is genuinely model-agnostic: it
    operates on raw frames BEFORE the model, using a conditional determinantal
    point process to pick `num_select` maximally-informative, minimally-redundant
    frames from a larger pool, conditioned on the question. Nothing about it
    depends on the LLM architecture, so the LLaVA-OV selector transfers verbatim —
    only the backbone that consumes the frames changes.

    Returns a list of PIL images of length num_select.
    """
    import logging
    import numpy as _np
    MDP3 = _load_mdp3_class()

    selector = MDP3("cuda")
    selector.n_selection = num_select
    selected = selector(pil_frames, question)

    if not selected or len(selected) < num_select:
        logging.warning("MDP3(Qwen3-VL): selector returned %d frames (expected %d) "
                        "— padding uniformly",
                        len(selected) if selected else 0, num_select)
        idxs = _np.linspace(0, len(pil_frames) - 1, num_select, dtype=int)
        selected = [pil_frames[i] for i in idxs]
    else:
        logging.warning("MDP3(Qwen3-VL) ACTIVE: pool=%d -> selected=%d frames",
                        len(pil_frames), len(selected))
    return selected


def load_model(model_path: str):
    """Load Qwen3-VL via transformers from_pretrained (no LLaVA)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return None, model, processor  # Qwen3 uses processor, not tokenizer + image_processor separately


# ---------------------------------------------------------------------------
# Video loading — standard subprocess-isolated
# ---------------------------------------------------------------------------
def load_frames(video_path: str, num_frames: int) -> list:
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
# Inference — Qwen3-VL native generate()
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(model, processor, frames: list, question: str,
                  num_frames: int = 32) -> str:
    """
    Qwen3-VL inference using native chat template + video preprocessing.
    Handles video via processor with temporal patch support.
    """
    # Build conversation using Qwen3's chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": question + POST_PROMPT},
            ],
        }
    ]

    # Apply chat template to build prompt
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process video + text together
    # CRITICAL: Qwen3VLVideoProcessor has do_sample_frames=True and fps=2, so by
    # default it RE-SAMPLES whatever frame list we hand it, ignoring --num_frames.
    # With no video_metadata it also warns "Defaulting to fps=24". We already
    # sampled exactly num_frames uniformly in load_video, so turn the processor's
    # own sampling OFF and let it consume our frames verbatim.
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames],
        return_tensors="pt",
        do_sample_frames=False,
    )
    # Report the frame count that ACTUALLY reaches the model, once, so a silent
    # re-sample can never go unnoticed again.
    if not getattr(run_inference, "_frames_logged", False):
        import logging
        pvg = inputs.get("pixel_values_videos", None)
        gt = inputs.get("video_grid_thw", None)
        logging.warning("Qwen3-VL FRAMES: requested=%d given=%d "
                        "pixel_values_videos=%s video_grid_thw=%s",
                        num_frames, len(frames),
                        tuple(pvg.shape) if pvg is not None else None,
                        gt.tolist() if gt is not None else None)
        run_inference._frames_logged = True

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=16,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


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
    parser.add_argument("--num_frames", type=int, default=32,
                        help="pool size sampled from the video (MDP3 selects from this)")
    parser.add_argument("--mdp3", action="store_true", help="apply MDP3 frame selection")
    parser.add_argument("--select_frames", type=int, default=8,
                        help="how many frames MDP3 keeps from the pool")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3-VL model...", flush=True)
    _, model, processor = load_model(args.model_path)

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
                # MDP3 acts BEFORE the model: pool -> conditional-DPP selection.
                if args.mdp3:
                    frames = select_frames_mdp3(frames, question, args.select_frames)
                prediction = run_inference(
                    model, processor, frames, question,
                    num_frames=len(frames),
                )
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}",
                      file=sys.stderr)
                prediction = ""
                torch.cuda.empty_cache()

        s = score_prediction(prediction, ground_truth)
        results.append({
            "idx": i,
            "video_path": sample["video_path"],
            "question_type": q_type,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": s,
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
        "accuracy": accuracy,
        "correct": correct,
        "total_scoreable": total,
        "total_na_skipped": na_count,
        "total_samples": len(results),
        "mdp3_params": {"enabled": bool(args.mdp3), "pool_frames": args.num_frames,
                        "select_frames": args.select_frames},
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "num_frames": args.num_frames,
        "note": "MDP3 on Qwen3-VL — conditional-DPP frame selection before the model",
        "per_category": per_category,
    }
    print(
        f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)",
        flush=True,
    )
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

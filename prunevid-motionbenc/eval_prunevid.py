#!/usr/bin/env python3
"""
PruneVid × MotionBench — thin wrapper.

Calls PruneVid's unmodified load_pllava / pllava_answer on MotionBench samples.
This file provides only: dataset iteration, subprocess video loading, NA-skip scoring.
All model code comes from the PruneVid repo unchanged.

Usage:
    PYTHONPATH=/project/rhu/dpalfaro/code/PruneVid:$PYTHONPATH \\
    python eval_prunevid.py \\
        --model_path ermu2001/pllava-7b \\
        --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        --output_dir /project/rhu/dpalfaro/results/prunevid_run1 \\
        [--num_frames 16] [--limit 50]
        [--cluster_ratio 0.5] [--temporal_segment_ratio 0.25]
        [--selected_layer 10] [--alpha 0.4] [--tau 0.8]
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
# Video loading — subprocess-isolated for NFS stale-handle safety
# ---------------------------------------------------------------------------

def load_frames(video_path: str, num_frames: int):
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr     = VideoReader(p, ctx=cpu(0))
            idx    = np.linspace(0, len(vr) - 1, n, dtype=int)
            frames = vr.get_batch(idx).asnumpy()
            q.put(("ok", frames))
        except Exception as e:
            q.put(("error", str(e)))

    q    = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frames, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill(); proc.join(timeout=5)
        raise RuntimeError(f"timeout (NFS stale?): {video_path}")
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill(); proc.join(timeout=5)
    if status == "error":
        raise RuntimeError(f"decode error: {video_path} — {data}")

    from PIL import Image
    return [Image.fromarray(f) for f in data]


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
    parser.add_argument("--model_path", required=True,
                        help="HF repo_id or local path, e.g. ermu2001/pllava-7b")
    parser.add_argument("--meta_path", required=True,
                        help="Path to video_info.meta.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames to sample per video (PLLaVA-7B native: 16)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N samples (smoke test)")

    # PruneVid pruning params — passed through to load_pllava unchanged
    parser.add_argument("--cluster_ratio",          type=float, default=0.5)
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25)
    parser.add_argument("--selected_layer",         type=int,   default=10)
    parser.add_argument("--alpha",                  type=float, default=0.4)
    parser.add_argument("--tau",                    type=float, default=0.8)
    parser.add_argument("--no_pruning", action="store_true", default=False,
                        help="Disable PruneVid — run vanilla PLLaVA-7B (baseline)")
    args = parser.parse_args()

    if args.no_pruning:
        args.cluster_ratio          = 1.0
        args.temporal_segment_ratio = 1.0

    os.makedirs(args.output_dir, exist_ok=True)

    # eval_utils.py has module-level imports for cv2 (needs libpng16), moviepy,
    # and matplotlib — none of which are available/usable in the prunevid env.
    # Stub them out before the deferred PruneVid imports below resolve eval_utils.
    import types as _types
    from importlib.machinery import ModuleSpec as _ModuleSpec
    for _m in ("cv2", "moviepy", "moviepy.editor",
               "matplotlib", "matplotlib.colors", "matplotlib.cm", "matplotlib.pyplot"):
        if _m not in sys.modules:
            _mod = _types.ModuleType(_m)
            _mod.__spec__ = _ModuleSpec(_m, None)  # importlib.util.find_spec rejects None __spec__
            sys.modules[_m] = _mod
    sys.modules["moviepy.editor"].VideoFileClip = None
    sys.modules["matplotlib.colors"].XKCD_COLORS = {}

    # mmcv.runner.load_checkpoint — used only for optical flow model weights.
    # mmcv can't be pip-installed (pkg_resources missing); provide a torch equivalent.
    if "mmcv" not in sys.modules:
        import torch as _torch
        def _load_checkpoint(model, filename, map_location="cpu", strict=False, **kw):
            ckpt = _torch.load(filename, map_location=map_location)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd, strict=strict)
            return ckpt
        _mmcv = _types.ModuleType("mmcv");           _mmcv.__spec__ = _ModuleSpec("mmcv", None)
        _mmcv_runner = _types.ModuleType("mmcv.runner"); _mmcv_runner.__spec__ = _ModuleSpec("mmcv.runner", None)
        _mmcv_runner.load_checkpoint = _load_checkpoint
        _mmcv.runner = _mmcv_runner
        sys.modules["mmcv"] = _mmcv
        sys.modules["mmcv.runner"] = _mmcv_runner

    from tasks.eval.model_utils import load_pllava, pllava_answer
    from tasks.eval.eval_utils import conv_eval_mvbench

    # pooling_shape temporal dim matches num_frames so all loaded frames contribute
    pooling_shape = (args.num_frames, 12, 12)

    print("Loading model...", flush=True)
    model, processor = load_pllava(
        args.model_path,
        num_frames=args.num_frames,
        use_lora=False,
        pooling_shape=pooling_shape,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        tau=args.tau,
        cluster_ratio=args.cluster_ratio,
        temporal_segment_ratio=args.temporal_segment_ratio,
    )

    # HF generate() strips media_type before calling prepare_inputs_for_generation.
    # Patch both prepare_inputs_for_generation (so it's in the inputs dict passed
    # to each forward call) and forward itself (force-assign, not setdefault, in
    # case the key is present but explicitly None).
    import functools as _functools
    _orig_pipg = model.prepare_inputs_for_generation
    def _pipg_video(*args, **kwargs):
        result = _orig_pipg(*args, **kwargs)
        result['media_type'] = 'video'
        return result
    model.prepare_inputs_for_generation = _pipg_video

    _orig_forward = model.forward
    @_functools.wraps(_orig_forward)
    def _forward_video(*args, **kwargs):
        kwargs['media_type'] = 'video'
        return _orig_forward(*args, **kwargs)
    model.forward = _forward_video

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
                frames = load_frames(video_path, args.num_frames)

                conv = conv_eval_mvbench.copy()
                conv.user_query(
                    question + POST_PROMPT,
                    is_mm=True,
                )
                conv.assistant_response(None)

                prediction, _ = pllava_answer(
                    conv=conv,
                    model=model,
                    processor=processor,
                    img_list=frames,
                    do_sample=False,
                    max_new_tokens=16,
                    temperature=1.0,
                )
            except Exception as e:
                import traceback
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                prediction = ""

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
        "pruning_enabled":  not args.no_pruning,
        "prunevid_params": {
            "cluster_ratio":          args.cluster_ratio,
            "temporal_segment_ratio": args.temporal_segment_ratio,
            "selected_layer":         args.selected_layer,
            "alpha":                  args.alpha,
            "tau":                    args.tau,
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

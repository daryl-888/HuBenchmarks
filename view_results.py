#!/usr/bin/env python3
"""
Results viewer for MotionBench native eval scripts.

Reads results.jsonl (our format) and video_info.meta.jsonl (original metadata)
and prints a clean summary: accuracy, per-sample table with question/choices/
prediction/video path, and scp commands to download the first N sample videos.

Usage:
    python view_results.py \\
        --results  /project/rhu/dpalfaro/results/sttm_llavavid_run1/results.jsonl \\
        --meta     /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \\
        [--n 10]   [--download_dir ~/demo_videos] [--carya_user dpalfaro]
"""

import argparse
import json
import os
import re
import sys

VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"


def find_video_path(video_path: str):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


def parse_choices(question_text: str):
    """Extract A/B/C/D choice lines from question text if present."""
    lines = question_text.strip().splitlines()
    choices = [l.strip() for l in lines if re.match(r"^[A-D][.)]\s", l.strip())]
    q_lines = [l for l in lines if not re.match(r"^[A-D][.)]\s", l.strip())]
    return "\n".join(q_lines).strip(), choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="results.jsonl from a native eval script")
    parser.add_argument("--meta", required=True,
                        help="video_info.meta.jsonl (MotionBench metadata)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of scoreable samples to display in detail")
    parser.add_argument("--download_dir", default="~/demo_videos",
                        help="Local directory for scp commands")
    parser.add_argument("--carya_user", default="dpalfaro",
                        help="Carya SSH username for scp commands")
    args = parser.parse_args()

    # Load metadata (question text + answer choices)
    meta = []
    with open(args.meta) as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))

    # Load results
    results = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    scoreable = [r for r in results if r["correct"] is not None]
    correct   = sum(r["correct"] for r in scoreable)
    total     = len(scoreable)
    na_count  = len(results) - total
    accuracy  = correct / total if total > 0 else 0.0

    # Per-category breakdown
    from collections import defaultdict
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)
    for r in scoreable:
        qt = r.get("question_type", "Unknown")
        cat_total[qt]   += 1
        cat_correct[qt] += r["correct"]

    label = os.path.basename(os.path.dirname(args.results))
    print("=" * 70)
    print(f"  {label}")
    print(f"  Accuracy : {correct}/{total} = {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  NA skipped: {na_count}   Total samples: {len(results)}")
    print()
    print(f"  {'Category':<35}  {'Correct':>7}  {'Total':>7}  {'Acc':>7}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}")
    for qt in sorted(cat_total.keys()):
        n  = cat_total[qt]
        c  = cat_correct[qt]
        a  = c / n if n > 0 else 0.0
        print(f"  {qt:<35}  {c:>7}  {n:>7}  {a*100:>6.2f}%")
    print("=" * 70)

    # Detailed table for first N scoreable samples
    shown = 0
    scp_commands = []

    for r in results:
        if r["correct"] is None:
            continue
        if shown >= args.n:
            break

        idx      = r["idx"]
        gt       = r["ground_truth"]
        pred     = r["prediction"]
        mark     = "CORRECT" if r["correct"] else "WRONG  "
        q_type   = r.get("question_type", "Unknown")
        vpath    = r["video_path"]

        # Get full question text from metadata
        if idx < len(meta):
            raw_q = meta[idx]["qa"][0].get("question", "")
            question, choices = parse_choices(raw_q)
        else:
            question, choices = "(metadata not found)", []

        full_path = find_video_path(vpath)
        if full_path is None:
            full_path = f"(not found on disk: {vpath})"

        print(f"\n[{mark}]  Sample {idx}  |  {q_type}")
        print(f"  Video    : {full_path}")
        print(f"  Question : {question}")
        if choices:
            for c in choices:
                print(f"             {c}")
        print(f"  GT       : {gt}     Predicted: {repr(pred)}")
        print("-" * 70)

        if full_path and full_path.startswith("/"):
            scp_commands.append(
                f"scp {args.carya_user}@carya.rcdc.uh.edu:{full_path} {args.download_dir}/"
            )
        shown += 1

    # scp commands for the first N sample videos
    if scp_commands:
        print(f"\n{'=' * 70}")
        print(f"  Download first {args.n} sample videos to {args.download_dir}")
        print(f"  Run on YOUR LOCAL MACHINE:")
        print(f"{'=' * 70}")
        print(f"  mkdir -p {args.download_dir}")
        for cmd in scp_commands:
            print(f"  {cmd}")
        print()


if __name__ == "__main__":
    main()

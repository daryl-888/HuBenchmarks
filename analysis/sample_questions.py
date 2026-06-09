#!/usr/bin/env python3
"""
Sample 5 questions from a results dir:
  - 3 wrong predictions, each from a different category
  - 2 correct predictions, each from a different category

Copies the videos to a staging dir and prints the scp command.

Usage (run on Carya):
    python sample_questions.py <results_dir> [--seed 42] [--stage_dir /project/rhu/dpalfaro/sample_videos]
"""
import argparse
import json
import os
import random
import shutil
import sys

VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"
DEFAULT_STAGE = "/project/rhu/dpalfaro/sample_videos"


def find_video(video_path):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


def pick(pool_by_cat, n, exclude_cats=()):
    cats = [c for c in pool_by_cat if pool_by_cat[c] and c not in exclude_cats]
    if len(cats) < n:
        cats = [c for c in pool_by_cat if pool_by_cat[c]]
    chosen_cats = random.sample(cats, min(n, len(cats)))
    return [random.choice(pool_by_cat[c]) for c in chosen_cats]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_dir", default=DEFAULT_STAGE)
    args = parser.parse_args()

    random.seed(args.seed)

    results = []
    with open(os.path.join(args.results_dir, "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    wrong_by_cat = {}
    right_by_cat = {}
    for r in results:
        if r["correct"] is None:
            continue
        cat = r.get("question_type", "Unknown")
        bucket = wrong_by_cat if r["correct"] == 0 else right_by_cat
        bucket.setdefault(cat, []).append(r)

    wrong_samples = pick(wrong_by_cat, 3)
    used_cats = {s["question_type"] for s in wrong_samples}
    right_samples = pick(right_by_cat, 2, exclude_cats=used_cats)

    selected = wrong_samples + right_samples

    os.makedirs(args.stage_dir, exist_ok=True)

    print(f"\nSampled from: {args.results_dir}")
    print("=" * 70)

    staged = []
    for i, s in enumerate(selected):
        label = "WRONG" if s["correct"] == 0 else "RIGHT"
        print(f"\n[{i+1}] {label} — {s.get('question_type', 'Unknown')}")
        print(f"  Video:      {s['video_path']}")
        print(f"  Truth:      {s['ground_truth']}")
        print(f"  Prediction: {s['prediction'] or '(empty)'}")

        src = find_video(s["video_path"])
        if src:
            dst = os.path.join(args.stage_dir, os.path.basename(s["video_path"]))
            shutil.copy2(src, dst)
            staged.append(dst)
            print(f"  Staged:     {dst}")
        else:
            print(f"  WARNING: video not found on disk")

    if staged:
        user = "dpalfaro"
        host = "carya.rcdc.uh.edu"
        remote_dir = args.stage_dir
        print(f"\n{'='*70}")
        print("SCP command (run on your local machine):")
        print(f"  scp {user}@{host}:\"{remote_dir}/*.mp4\" .")
        print("\nOr individual files:")
        for dst in staged:
            print(f"  scp {user}@{host}:\"{dst}\" .")


if __name__ == "__main__":
    main()

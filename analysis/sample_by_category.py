#!/usr/bin/env python3
"""
Sample N examples from specific categories in a MotionBench results dir.

Supports custom format (results.jsonl) used by PruneVid, HoliTom, FlashVID, VisionZip.

Usage (run on Carya):
    python sample_by_category.py <results_dir> --cats "Location Related Motion" "Action Order"
    python sample_by_category.py <results_dir> --list-cats            # show all available categories
    python sample_by_category.py <results_dir> --cats "location" "action order" -n 3 --seed 0

Category matching is case-insensitive substring search.
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

VIDEO_BASE    = "/project/rhu/MotionBench_Data/MotionBench"
DEFAULT_META  = "/project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl"
DEFAULT_STAGE = "/project/rhu/dpalfaro/sample_videos"


def find_video(video_path):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


def load_results(results_dir):
    meta_by_idx = {}
    with open(DEFAULT_META) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                meta_by_idx[i] = json.loads(line)

    records = []
    with open(os.path.join(results_dir, "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            m = meta_by_idx.get(r["idx"], {})
            qa = m.get("qa", [{}])[0]
            records.append({
                "idx":           r["idx"],
                "video_path":    r["video_path"],
                "question_type": r.get("question_type", "Unknown"),
                "ground_truth":  r["ground_truth"],
                "prediction":    r.get("prediction", ""),
                "correct":       r["correct"],
                "question":      qa.get("question", "(not found)"),
                "options":       qa.get("options", {}),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--cats", nargs="+", default=["Location Related Motion", "Action Order"],
                        help="Category substrings to sample (case-insensitive)")
    parser.add_argument("-n", type=int, default=3, help="Samples per category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_dir", default=DEFAULT_STAGE)
    parser.add_argument("--list-cats", action="store_true", help="Print all categories and exit")
    args = parser.parse_args()

    random.seed(args.seed)
    records = load_results(args.results_dir)

    if args.list_cats:
        counts = defaultdict(int)
        for r in records:
            if r["correct"] is not None:
                counts[r["question_type"]] += 1
        print(f"\nCategories in {args.results_dir}:")
        for cat in sorted(counts):
            print(f"  {cat:<45} {counts[cat]:>4} samples")
        return

    by_cat = defaultdict(list)
    for r in records:
        if r["correct"] is not None:
            by_cat[r["question_type"]].append(r)

    os.makedirs(args.stage_dir, exist_ok=True)
    staged = []
    any_found = False

    for query in args.cats:
        matched = [cat for cat in by_cat if query.lower() in cat.lower()]
        if not matched:
            print(f"\n[WARNING] No category matching '{query}'. Use --list-cats to see all.")
            continue

        for cat in matched:
            pool = by_cat[cat]
            chosen = random.sample(pool, min(args.n, len(pool)))
            any_found = True

            print(f"\n{'='*70}")
            print(f"  CATEGORY: {cat}  ({len(pool)} total samples)")
            print(f"  Showing {len(chosen)} of {len(pool)}")
            print(f"{'='*70}")

            for i, s in enumerate(chosen):
                label = "CORRECT" if s["correct"] == 1 else "WRONG"
                print(f"\n  [{i+1}] {label}")
                print(f"  Video:      {s['video_path']}")
                print(f"  Question:   {s['question']}")
                if s["options"]:
                    for k, v in sorted(s["options"].items()):
                        print(f"              {k}) {v}")
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
        print(f"\n{'='*70}")
        print("SCP command (run on your local machine):")
        print(f"  scp dpalfaro@carya.rcdc.uh.edu:\"{args.stage_dir}/*.mp4\" .")
        print("\nIndividual files:")
        for dst in staged:
            print(f"  scp dpalfaro@carya.rcdc.uh.edu:\"{dst}\" .")


if __name__ == "__main__":
    main()

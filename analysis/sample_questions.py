#!/usr/bin/env python3
"""
Sample 5 questions from a results dir:
  - 3 wrong predictions, each from a different category
  - 2 correct predictions, each from a different category

Copies the videos to a staging dir and prints the scp command.

Supports:
  - Custom format (results.jsonl): FlashVID, PruneVid, HoliTom, VisionZip
  - lmms_eval format (*/motionbench.json with 'logs'): DyCoke, STTM, VideoITG
  - New lmms_eval format (*_samples_motionbench.jsonl): FastVID

Usage (run on Carya):
    python sample_questions.py <results_dir> [--seed 42] [--stage_dir /project/rhu/dpalfaro/sample_videos]
"""
import argparse
import glob
import json
import os
import random
import shutil

VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"
DEFAULT_STAGE = "/project/rhu/dpalfaro/sample_videos"
DEFAULT_META = "/project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl"


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


def load_custom(results_dir):
    records = []
    meta_by_idx = {}
    meta_path = DEFAULT_META
    with open(meta_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                meta_by_idx[i] = json.loads(line)

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
                "question":      qa.get("question", "(question not found)"),
            })
    return records


def load_lmms_eval(results_dir):
    matches = glob.glob(os.path.join(results_dir, "*", "motionbench.json"))
    if not matches:
        raise FileNotFoundError(f"No motionbench.json found under {results_dir}")
    with open(matches[0]) as f:
        d = json.load(f)

    records = []
    for log in d["logs"]:
        doc = log.get("doc", {})
        resps = log.get("filtered_resps", [])
        if resps and isinstance(resps[0], list):
            pred = resps[0][0]
        elif resps:
            pred = resps[0]
        else:
            pred = ""
        qa = doc.get("qa", [{}])[0]
        records.append({
            "idx":           log.get("doc_id", 0),
            "video_path":    doc.get("video_path", ""),
            "question_type": log.get("category", doc.get("question_type", "Unknown")),
            "ground_truth":  log.get("target", ""),
            "prediction":    pred,
            "correct":       log.get("motionbench_accuracy"),
            "question":      qa.get("question", "(question not found)"),
        })
    return records


def _extract_pred(resps):
    """Extract prediction string from filtered_resps regardless of nesting."""
    if not resps:
        return ""
    r = resps[0]
    if isinstance(r, (list, tuple)):
        return str(r[0]) if r else ""
    return str(r)


def load_lmms_samples(results_dir):
    """New lmms_eval format: timestamped *_samples_motionbench.jsonl."""
    matches = glob.glob(os.path.join(results_dir, "*", "*_samples_motionbench.jsonl"))
    if not matches:
        matches = glob.glob(os.path.join(results_dir, "*_samples_motionbench.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No *_samples_motionbench.jsonl found under {results_dir}")

    # Load full metadata to recover questions (full_docs:false strips doc fields)
    meta_by_path = {}
    try:
        with open(DEFAULT_META) as f:
            for line in f:
                line = line.strip()
                if line:
                    m = json.loads(line)
                    meta_by_path[m.get("video_path", "")] = m
    except Exception:
        pass

    records = []
    with open(matches[0]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = log.get("doc") or {}
            video_path = doc.get("video_path", "")
            pred = _extract_pred(log.get("filtered_resps") or [])
            acc = log.get("motionbench_accuracy")
            cat = log.get("category") or doc.get("question_type", "Unknown")

            # Recover question from metadata if doc is stripped
            qa_list = doc.get("qa") or []
            question = qa_list[0].get("question", "") if qa_list else ""
            if not question and video_path in meta_by_path:
                meta_qa = meta_by_path[video_path].get("qa", [{}])
                question = meta_qa[0].get("question", "(question not found)") if meta_qa else "(question not found)"
            if not question:
                question = "(question not found)"

            records.append({
                "idx":           log.get("doc_id", 0),
                "video_path":    video_path,
                "question_type": cat,
                "ground_truth":  log.get("target", ""),
                "prediction":    pred,
                "correct":       acc,
                "question":      question,
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_dir", default=DEFAULT_STAGE)
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"\nSampled from: {args.results_dir}")

    results_jsonl = os.path.join(args.results_dir, "results.jsonl")
    samples_glob = (glob.glob(os.path.join(args.results_dir, "*", "*_samples_motionbench.jsonl")) or
                    glob.glob(os.path.join(args.results_dir, "*_samples_motionbench.jsonl")))
    if os.path.exists(results_jsonl):
        results = load_custom(args.results_dir)
    elif samples_glob:
        results = load_lmms_samples(args.results_dir)
    else:
        results = load_lmms_eval(args.results_dir)

    wrong_by_cat = {}
    right_by_cat = {}
    for r in results:
        if r["correct"] is None:
            continue
        cat = r["question_type"]
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
        print(f"\n[{i+1}] {label} — {s['question_type']}")
        print(f"  Video:      {s['video_path']}")
        print(f"  Question:   {s['question']}")
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
        print(f"\n{'='*70}")
        print("SCP command (run on your local machine):")
        print(f"  scp {user}@{host}:\"{args.stage_dir}/*.mp4\" .")
        print("\nOr individual files:")
        for dst in staged:
            print(f"  scp {user}@{host}:\"{dst}\" .")


if __name__ == "__main__":
    main()

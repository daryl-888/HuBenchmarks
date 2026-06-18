#!/usr/bin/env python3
"""
Per-category accuracy breakdown for MotionBench results.

Usage:
    python analyze_results.py <results_dir> [<results_dir2> ...]

Supports:
  - Custom format (results.jsonl + summary.json): FlashVID, PruneVid, HoliTom
  - lmms_eval format (*/motionbench.json with 'logs' list): DyCoke, STTM, VideoITG
"""
import glob
import json
import os
import sys
from collections import defaultdict


def _find_lmms_file(results_dir):
    matches = glob.glob(os.path.join(results_dir, "*", "motionbench.json"))
    return matches[0] if matches else None


def _print_table(title, accuracy, correct, total_scoreable, na_skipped, by_cat, extra=""):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  Overall:   {accuracy:.2%}  "
          f"({correct}/{total_scoreable} scoreable, {na_skipped} NA skipped)")
    if extra:
        print(extra)
    print(f"\n  {'Category':<42} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print(f"  {'-'*42} {'-'*7} {'-'*7} {'-'*9}")
    for cat in sorted(by_cat, key=lambda c: by_cat[c]["correct"] / max(by_cat[c]["total"], 1),
                      reverse=True):
        c = by_cat[cat]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0.0
        print(f"  {cat:<42} {c['correct']:>7} {c['total']:>7} {acc:>9.1%}")
    print()


def analyze_custom(results_dir):
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)
    results = []
    with open(os.path.join(results_dir, "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["correct"] is None:
            continue
        cat = r.get("question_type", "Unknown")
        by_cat[cat]["total"] += 1
        by_cat[cat]["correct"] += r["correct"]

    extra_lines = []
    if "model" in summary:
        extra_lines.append(f"  Model:     {os.path.basename(summary['model'])}")
    if "pruning_enabled" in summary:
        p = summary.get("prunevid_params", {})
        tag = "enabled" if summary["pruning_enabled"] else "disabled (baseline)"
        extra_lines.append(f"  Pruning:   {tag}")
        if summary["pruning_enabled"]:
            extra_lines.append(
                f"             cluster_ratio={p.get('cluster_ratio')}  "
                f"temporal_segment_ratio={p.get('temporal_segment_ratio')}  "
                f"selected_layer={p.get('selected_layer')}  "
                f"alpha={p.get('alpha')}  tau={p.get('tau')}"
            )
    if "holitom_params" in summary:
        h = summary["holitom_params"]
        extra_lines.append(
            f"  HoliTom:   RETAIN_RATIO={h.get('RETAIN_RATIO')}  "
            f"T={h.get('T')}  k={h.get('HOLITOM_k')}  r={h.get('HOLITOM_r')}"
        )
    if "flashvid_params" in summary:
        fv = summary["flashvid_params"]
        extra_lines.append(
            f"  FlashVID:  retention_ratio={fv.get('retention_ratio')}  "
            f"alpha={fv.get('alpha')}  temporal_threshold={fv.get('temporal_threshold')}  "
            f"num_frames={fv.get('num_frames')}"
        )

    _print_table(
        title=os.path.basename(results_dir.rstrip("/")),
        accuracy=summary["accuracy"],
        correct=summary["correct"],
        total_scoreable=summary["total_scoreable"],
        na_skipped=summary["total_na_skipped"],
        by_cat=by_cat,
        extra="\n".join(extra_lines),
    )


def analyze_lmms_eval(results_dir):
    lmms_file = _find_lmms_file(results_dir)
    if not lmms_file:
        raise FileNotFoundError(f"No motionbench.json found under {results_dir}")

    with open(lmms_file) as f:
        d = json.load(f)

    logs = d["logs"]
    model_args = d.get("args", {}).get("model_args", "")

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    total_scoreable = 0
    na_skipped = 0

    for log in logs:
        acc = log.get("motionbench_accuracy")
        cat = log.get("category", "Unknown")
        if acc is None:
            na_skipped += 1
            continue
        by_cat[cat]["total"] += 1
        by_cat[cat]["correct"] += acc
        total_scoreable += 1
        correct += acc

    accuracy = correct / total_scoreable if total_scoreable > 0 else 0.0
    extra = f"  Model args: {model_args}" if model_args else ""

    _print_table(
        title=os.path.basename(results_dir.rstrip("/")),
        accuracy=accuracy,
        correct=correct,
        total_scoreable=total_scoreable,
        na_skipped=na_skipped,
        by_cat=by_cat,
        extra=extra,
    )


def analyze(results_dir):
    if (os.path.exists(os.path.join(results_dir, "results.jsonl")) and
            os.path.exists(os.path.join(results_dir, "summary.json"))):
        analyze_custom(results_dir)
    else:
        analyze_lmms_eval(results_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_dir> [...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        analyze(d)

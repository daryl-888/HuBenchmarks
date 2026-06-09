#!/usr/bin/env python3
"""
Per-category accuracy breakdown for MotionBench results.

Usage:
    python analyze_results.py <results_dir> [<results_dir2> ...]

Each results_dir must contain results.jsonl and summary.json.
"""
import json
import os
import sys
from collections import defaultdict


def analyze(results_dir):
    summary_path = os.path.join(results_dir, "summary.json")
    results_path = os.path.join(results_dir, "results.jsonl")

    with open(summary_path) as f:
        summary = json.load(f)

    results = []
    with open(results_path) as f:
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

    print(f"\n{'='*65}")
    print(f"  {os.path.basename(results_dir.rstrip('/'))}")
    print(f"{'='*65}")
    print(f"  Overall:   {summary['accuracy']:.2%}  "
          f"({summary['correct']}/{summary['total_scoreable']} scoreable, "
          f"{summary['total_na_skipped']} NA skipped)")

    # Model / run details
    if "model" in summary:
        print(f"  Model:     {os.path.basename(summary['model'])}")
    if "pruning_enabled" in summary:
        p = summary.get("prunevid_params", {})
        tag = "enabled" if summary["pruning_enabled"] else "disabled (baseline)"
        print(f"  Pruning:   {tag}")
        if summary["pruning_enabled"]:
            print(f"             cluster_ratio={p.get('cluster_ratio')}  "
                  f"temporal_segment_ratio={p.get('temporal_segment_ratio')}  "
                  f"selected_layer={p.get('selected_layer')}  "
                  f"alpha={p.get('alpha')}  tau={p.get('tau')}")
    if "holitom_params" in summary:
        h = summary["holitom_params"]
        print(f"  HoliTom:   RETAIN_RATIO={h.get('RETAIN_RATIO')}  "
              f"T={h.get('T')}  k={h.get('HOLITOM_k')}  r={h.get('HOLITOM_r')}")
    if "grounding_jsonl" in summary:
        print(f"  Grounding: {os.path.basename(summary['grounding_jsonl'])}")

    print(f"\n  {'Category':<42} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print(f"  {'-'*42} {'-'*7} {'-'*7} {'-'*9}")

    for cat in sorted(by_cat, key=lambda c: by_cat[c]["correct"] / max(by_cat[c]["total"], 1),
                      reverse=True):
        c = by_cat[cat]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0.0
        print(f"  {cat:<42} {c['correct']:>7} {c['total']:>7} {acc:>9.1%}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_dir> [...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        analyze(d)

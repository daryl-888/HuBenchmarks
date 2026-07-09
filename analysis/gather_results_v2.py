#!/usr/bin/env python3
"""
Results gatherer for the reorganized HuBenchMarks structure.

Scans ovqwen/, ovqwen2/, ovqwen3/, and other_backbones/ for model directories,
reads their summary.json files, and produces a consolidated results table.

Usage:
    python analysis/gather_results_v2.py [--json] [--csv output.csv]

Categories (MotionBench):
    Action Order, Camera Motion, Location-related Motion,
    Motion Recognition, Motion-related Objects, Repetition Count
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

FOLDERS = ["ovqwen", "ovqwen2", "ovqwen3", "other_backbones"]
CATEGORIES = [
    "Action Order",
    "Camera Motion",
    "Location-related Motion",
    "Motion Recognition",
    "Motion-related Objects",
    "Repetition Count",
]

BACKBONE_MAP = {
    "ovqwen": "Qwen 1.5 (llava-ov-7b)",
    "ovqwen2": "Qwen2 (llava-ov-7b-qwen2)",
    "ovqwen3": "Qwen3 (template)",
    "other_backbones": "Various",
}


def find_models(root: str):
    """Find all model directories that have a summary.json."""
    models = []
    base = Path(root)
    if not base.is_dir():
        return models
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        summary_path = entry / "summary.json"
        results_path = entry / "results.jsonl"
        if summary_path.exists() and results_path.exists():
            models.append(str(entry))
    return models


def gather_results(root_dir: str):
    """Gather results from all folders."""
    all_results = {}
    for folder in FOLDERS:
        folder_path = Path(root_dir) / folder
        if not folder_path.is_dir():
            continue
        backbone = BACKBONE_MAP.get(folder, "Unknown")
        for model_entry in sorted(folder_path.iterdir()):
            if not model_entry.is_dir():
                continue
            summary_path = model_entry / "summary.json"
            if not summary_path.exists():
                continue
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            model_name = model_entry.name.replace("-motionbenc", "").replace("_NOG", " (NOG)")
            is_ported = "PORTED.md" in [p.name for p in model_entry.iterdir()] or "_NOG" in model_entry.name

            per_cat = summary.get("per_category", {})
            results_path = model_entry / "results.jsonl"
            has_results = results_path.exists()

            all_results[model_entry.name] = {
                "folder": folder,
                "model": model_name,
                "backbone": backbone,
                "accuracy": summary.get("accuracy"),
                "correct": summary.get("correct", 0),
                "total": summary.get("total_scoreable", 0),
                "na_skipped": summary.get("total_na_skipped", 0),
                "per_category": per_cat,
                "is_ported": is_ported,
                "has_results": has_results,
                "model_path": summary.get("model", ""),
                "frames": summary.get("num_frames", "?"),
            }
    return all_results


def print_table(results: dict):
    """Print a formatted results table."""
    sorted_models = sorted(
        results.values(),
        key=lambda x: (x["accuracy"] if x["accuracy"] is not None else -1),
        reverse=True,
    )

    header = f"{'Model':<30} {'Backbone':<30} {'Accuracy':>10} {'Correct/Total':>15} {'NA':>6} {'Frames':>7} {'Status':<12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for m in sorted_models:
        if m["accuracy"] is not None:
            acc_str = f"{m['accuracy']*100:.2f}%"
            correct_str = f"{m['correct']}/{m['total']}"
        else:
            acc_str = "—"
            correct_str = "—"

        na_str = str(m["na_skipped"]) if m["na_skipped"] else "—"
        frames_str = str(m["frames"]) if m["frames"] else "—"

        if m["accuracy"] is not None:
            status = "✅"
        elif m["has_results"]:
            status = "⏳"
        else:
            status = "⬜"

        if m["is_ported"]:
            status += " ported"

        print(f"{m['model']:<30} {m['backbone']:<30} {acc_str:>10} {correct_str:>15} {na_str:>6} {frames_str:>7} {status:<12}")

    print(sep)
    print(f"Total: {len(sorted_models)} models")


def print_per_category(results: dict):
    """Print per-category breakdown for models with results."""
    print(f"\n{'='*90}")
    print("  Per-Category Breakdown")
    print(f"{'='*90}")

    for model_name, m in sorted(results.items()):
        per_cat = m.get("per_category", {})
        if not per_cat:
            continue
        acc = m["accuracy"]
        acc_str = f"{acc*100:.2f}%" if acc is not None else "—"

        print(f"\n  {m['model']} ({m['backbone']}) — {acc_str}")
        print(f"  {'Category':<35} {'Correct':>7} {'Total':>7} {'Acc':>9}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*9}")

        for cat in CATEGORIES:
            c = per_cat.get(cat, {})
            correct = c.get("correct", 0)
            total = c.get("total", 0)
            cat_acc = correct / total if total > 0 else 0.0
            print(f"  {cat:<35} {correct:>7} {total:>7} {cat_acc:>8.1%}")

        # Any extra categories not in the standard list
        for cat in sorted(per_cat.keys()):
            if cat in CATEGORIES:
                continue
            c = per_cat[cat]
            correct = c.get("correct", 0)
            total = c.get("total", 0)
            cat_acc = correct / total if total > 0 else 0.0
            print(f"  {cat:<35} {correct:>7} {total:>7} {cat_acc:>8.1%}")


def export_csv(results: dict, path: str):
    """Export results to CSV."""
    with open(path, "w") as f:
        f.write("model,folder,backbone,accuracy,correct,total_scoreable,na_skipped,frames,is_ported")
        for cat in CATEGORIES:
            f.write(f",cat_{cat.replace(' ', '_')}")
        f.write("\n")

        for model_name, m in sorted(results.items()):
            acc = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else ""
            f.write(f"{m['model']},{m['folder']},{m['backbone']},{acc},{m['correct']},{m['total']},{m['na_skipped']},{m['frames']},{m['is_ported']}")
            per_cat = m.get("per_category", {})
            for cat in CATEGORIES:
                c = per_cat.get(cat, {})
                cat_acc = c.get("correct", 0) / c.get("total", 1) if c.get("total") else ""
                f.write(f",{cat_acc}")
            f.write("\n")

    print(f"\nCSV exported to {path}")


def export_json(results: dict, path: str):
    """Export results to JSON."""
    with open(path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_category"} for k, v in results.items()}, f, indent=2)
    print(f"JSON exported to {path}")


def main():
    parser = argparse.ArgumentParser(description="Gather MotionBench results across all backbone folders")
    parser.add_argument("--json", action="store_true", help="Export JSON")
    parser.add_argument("--csv", type=str, help="Export CSV to file")
    parser.add_argument("--per-category", action="store_true", help="Show per-category breakdown")
    parser.add_argument("--root", default=".", help="Root directory (default: current)")
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    results = gather_results(str(root_dir))

    if not results:
        print("No results found. Make sure you're in the repo root.", file=sys.stderr)
        sys.exit(1)

    print_table(results)

    if args.per_category:
        print_per_category(results)

    if args.json:
        export_json(results, str(root_dir / "analysis" / "results.json"))

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()

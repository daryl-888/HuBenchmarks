#!/usr/bin/env python3
"""
show_examples.py — pull qualitative examples out of a gated run.

Every number in this project is an aggregate. This prints the underlying
samples: the video, the question with its options, the ground truth, and what
the model actually answered. Default is 3 incorrect + 2 correct, which is enough
to sanity-check that a score reflects real behaviour rather than a formatting
artefact (the VisionZip 0% and the caption-emitting DyTo run would both have
been obvious in one glance at this output).

The run's results.jsonl has no question text — only video_path, ground_truth,
prediction and correct. The question lives in MotionBench's metadata, so this
joins the two on video_path.

Usage
-----
  python3 scripts/show_examples.py <run_dir> [--wrong N] [--right N]
                                   [--category "Repetition Count"]
                                   [--meta PATH] [--videos PATH]
                                   [--seed N] [--json]

Examples
--------
  # 3 wrong + 2 right from the DyTo run
  python3 scripts/show_examples.py $HUVLLM_RESULTS/ob_dyto_run

  # where a method loses to the backbone, in its weakest category
  python3 scripts/show_examples.py $HUVLLM_RESULTS/w3_sttm_run \\
      --category "Repetition Count" --wrong 5 --right 0

  # machine-readable, e.g. to hand to a viewer
  python3 scripts/show_examples.py $HUVLLM_RESULTS/w3_aim_run --json
"""
import argparse
import json
import os
import random
import sys
import textwrap

# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH",
                            "/project/rhu/MotionBench_Data/MotionBench")
DEFAULT_META = os.path.join(VIDEO_BASE, "video_info.meta.jsonl")


def load_meta(path):
    """video_path -> (question, answer, question_type). One entry per video."""
    meta = {}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            qa = (d.get("qa") or [{}])[0]
            meta.setdefault(d.get("video_path"), (
                qa.get("question", ""), qa.get("answer", ""),
                d.get("question_type", "")))
    return meta


def resolve_video(video_path):
    """MotionBench splits videos across two subdirs; return the real path."""
    for sub in ("self-collected", "public-dataset"):
        p = os.path.join(VIDEO_BASE, sub, video_path)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Show sample videos + questions from a run.")
    ap.add_argument("run_dir", help="a results dir containing results.jsonl")
    ap.add_argument("--wrong", type=int, default=3, help="incorrect examples (default 3)")
    ap.add_argument("--right", type=int, default=2, help="correct examples (default 2)")
    ap.add_argument("--category", help='e.g. "Repetition Count"')
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--videos", help="override $MOTIONBENCH video root")
    ap.add_argument("--seed", type=int, default=0, help="0 = deterministic pick")
    ap.add_argument("--json", action="store_true", help="emit JSON, not text")
    args = ap.parse_args()

    if args.videos:
        global VIDEO_BASE
        VIDEO_BASE = args.videos

    res = os.path.join(args.run_dir, "results.jsonl")
    if not os.path.exists(res):
        sys.exit(f"no results.jsonl in {args.run_dir}")
    if not os.path.exists(args.meta):
        sys.exit(f"metadata not found: {args.meta} (set $MOTIONBENCH)")

    rows = []
    with open(res) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # NA samples (correct is None) are unanswerable and excluded from scoring,
    # so they are not informative as examples either.
    rows = [r for r in rows if r.get("correct") is not None]
    if args.category:
        rows = [r for r in rows if r.get("question_type") == args.category]
        if not rows:
            sys.exit(f"no scoreable samples in category {args.category!r}")

    meta = load_meta(args.meta)
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]

    rnd = random.Random(args.seed)
    pick = (rnd.sample(wrong, min(args.wrong, len(wrong)))
            + rnd.sample(right, min(args.right, len(right))))

    acc = (100.0 * len(right) / len(rows)) if rows else 0.0
    out = []
    for r in pick:
        vp = r.get("video_path", "")
        q, ans, _ = meta.get(vp, ("", "", ""))
        out.append({
            "idx": r.get("idx"),
            "correct": bool(r["correct"]),
            "category": r.get("question_type"),
            "video": vp,
            "video_full_path": resolve_video(vp),
            "question": q,
            "ground_truth": r.get("ground_truth") or ans,
            "prediction": r.get("prediction"),
        })

    if args.json:
        print(json.dumps({"run": args.run_dir, "accuracy_pct": round(acc, 2),
                          "n_scoreable": len(rows), "examples": out}, indent=2))
        return

    print(f"\n{os.path.basename(args.run_dir)} — {acc:.2f}% "
          f"({len(right)}/{len(rows)} scoreable"
          + (f", category={args.category}" if args.category else "") + ")")
    if args.seed == 0:
        print("seed=0 (deterministic — same samples every run; use --seed N to vary)")
    print("=" * 74)
    for e in out:
        mark = "✓ CORRECT" if e["correct"] else "✗ WRONG"
        print(f"\n[{mark}]  idx={e['idx']}  {e['category']}")
        print(f"  video : {e['video']}")
        if e["video_full_path"]:
            print(f"  path  : {e['video_full_path']}")
        else:
            print("  path  : NOT FOUND on disk (missing/relocated video)")
        first = True
        for ln in (e["question"] or "(question not found in metadata)").split("\n"):
            for w in textwrap.wrap(ln, 68) or [""]:
                print(f"  {'question:' if first else '         '} {w}"
                      if first else f"           {w}")
                first = False
        print(f"  truth : {e['ground_truth']}    model: {e['prediction']}")
    print()


if __name__ == "__main__":
    main()

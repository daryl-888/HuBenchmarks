#!/usr/bin/env python3
"""
check_run.py — sanity gate for a completed MotionBench eval run.

Turns *silent-wrong* results into a loud FAIL. Every check here corresponds to a
real silent failure this project has already been burned by:

  * FastV "enabled: false"      -> pruning hook never fired; ran as pure baseline
  * VisionZip 0%                -> empty / degenerate predictions, still "scored"
  * PruneVID 44%                -> weights silently dropped -> near-random output
  * no-checkpoint truncation    -> job died mid-run, partial results.jsonl
  * dataset drift               -> NA count != ~4034 means wrong metadata loaded

It reads the two files every eval script writes:
    <run_dir>/summary.json      (accuracy, correct, total_scoreable, ...)
    <run_dir>/results.jsonl     (one row per sample: prediction, ground_truth, correct)

Usage:
    python scripts/check_run.py <run_dir> [--expect-method NAME] [--strict]
    python scripts/check_run.py /project/rhu/dpalfaro/results/stage2_aim_baseline

Exit code 0 = all gates pass. Non-zero = at least one FAIL (use in sbatch to
refuse to record a bad run). --strict also fails on WARN.

Designed to run on Carya right after the eval, or locally after pulling a
summary. Pure stdlib, no deps.
"""

import argparse
import json
import os
import sys
from collections import Counter

# MotionBench ground truth (from CLAUDE.md / dataset-verification).
EXPECTED_TOTAL_SAMPLES = 8052
EXPECTED_NA = 4034          # answer == "NA", skipped in scoring
EXPECTED_SCOREABLE = 4018   # 8052 - 4034
NA_TOLERANCE = 50           # a few videos legitimately skip (NFS black-frame)

# Accuracy sanity band for a 4-way MCQ. Below the floor => something is broken
# (random is ~25%, but a real-but-degraded model still lands ~40%+; anything
# under FLOOR has historically meant dropped weights or empty predictions).
ACC_FLOOR = 0.40
ACC_CEIL = 0.95             # nobody is at 95% on MotionBench; that's a bug/leak.

# If a single predicted letter dominates this share of answers, the model
# probably collapsed to a constant (VisionZip-style degeneration).
DEGENERATE_SHARE = 0.85

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = RESET = ""


class Gate:
    def __init__(self):
        self.fails = []
        self.warns = []
        self.oks = []

    def ok(self, msg):
        self.oks.append(msg)
        print(f"{GREEN}[ OK ]{RESET} {msg}")

    def warn(self, msg):
        self.warns.append(msg)
        print(f"{YELLOW}[WARN]{RESET} {msg}")

    def fail(self, msg):
        self.fails.append(msg)
        print(f"{RED}[FAIL]{RESET} {msg}")


def load_summary(run_dir, g):
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        g.fail(f"no summary.json in {run_dir} — job likely died before finishing")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        g.fail(f"summary.json unreadable ({e}) — truncated write / dead job")
        return None


def iter_results(run_dir, g):
    """Yield parsed rows from results.jsonl; empty list (and a FAIL) if absent."""
    path = os.path.join(run_dir, "results.jsonl")
    if not os.path.exists(path):
        g.fail("no results.jsonl — cannot verify predictions")
        return []
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                g.warn(f"results.jsonl line {ln} is not valid JSON (partial write?)")
    return rows


def check_completeness(summary, rows, g):
    n = summary.get("total_samples", len(rows))
    if n != EXPECTED_TOTAL_SAMPLES:
        g.fail(f"total_samples={n}, expected {EXPECTED_TOTAL_SAMPLES} "
               f"— run truncated (no checkpointing: it restarts from scratch)")
    else:
        g.ok(f"all {n} samples present")

    if rows and len(rows) != n:
        g.warn(f"summary says {n} samples but results.jsonl has {len(rows)} rows")


def check_na_accounting(summary, g):
    na = summary.get("total_na_skipped")
    if na is None:
        g.warn("summary has no total_na_skipped field — cannot verify NA accounting")
        return
    if abs(na - EXPECTED_NA) > NA_TOLERANCE:
        g.fail(f"NA skipped={na}, expected ~{EXPECTED_NA} "
               f"— wrong metadata loaded or scoring protocol drifted")
    else:
        g.ok(f"NA accounting sane ({na} skipped, expected ~{EXPECTED_NA})")


def check_accuracy_band(summary, g):
    acc = summary.get("accuracy")
    total = summary.get("total_scoreable", 0)
    if acc is None:
        g.fail("summary has no accuracy field")
        return
    if total == 0:
        g.fail("total_scoreable=0 — nothing was scored (all predictions NA/empty?)")
        return
    if acc < ACC_FLOOR:
        g.fail(f"accuracy={acc:.4f} below floor {ACC_FLOOR} "
               f"— dropped weights / empty predictions (PruneVID-44% signature)")
    elif acc > ACC_CEIL:
        g.fail(f"accuracy={acc:.4f} above ceiling {ACC_CEIL} "
               f"— too good to be true; label/leak bug")
    else:
        g.ok(f"accuracy {acc:.4f} in sane band [{ACC_FLOOR}, {ACC_CEIL}]")


def check_predictions_nondegenerate(rows, g):
    """Empty or constant predictions = VisionZip-0% signature."""
    if not rows:
        return
    preds = [str(r.get("prediction", "")).strip() for r in rows]
    empty = sum(1 for p in preds if p == "")
    empty_share = empty / len(preds)
    if empty_share > 0.20:
        g.fail(f"{empty_share:.0%} of predictions are empty "
               f"— model produced no output (VisionZip-0% signature)")
    elif empty_share > 0.02:
        g.warn(f"{empty_share:.0%} of predictions are empty")
    else:
        g.ok(f"predictions non-empty ({empty} empty / {len(preds)})")

    # Collapse to a single letter?
    letters = [p.upper()[0] for p in preds if p and p[0].upper() in "ABCD"]
    if letters:
        top_letter, top_count = Counter(letters).most_common(1)[0]
        share = top_count / len(letters)
        if share > DEGENERATE_SHARE:
            g.fail(f"prediction '{top_letter}' is {share:.0%} of all answers "
                   f"— model collapsed to a constant")
        else:
            g.ok(f"prediction distribution looks real "
                 f"(top letter '{top_letter}' = {share:.0%})")


def check_method_engaged(summary, expect_method, g):
    """
    FastV-stub signature: a *_params block whose 'enabled' is false, or that
    otherwise shows the method was a no-op. This is the check that would have
    caught FastV running as a plain baseline while labelled 'FastV'.
    """
    param_blocks = {k: v for k, v in summary.items()
                    if k.endswith("_params") and isinstance(v, dict)}
    if not param_blocks:
        if expect_method:
            g.warn(f"no *_params block in summary — cannot confirm {expect_method} "
                   f"actually engaged")
        return
    for name, params in param_blocks.items():
        method = name[:-len("_params")]
        if params.get("enabled") is False:
            g.fail(f"{method}_params.enabled == false "
                   f"— method did NOT run; this is a mislabelled baseline "
                   f"(FastV-stub signature)")
        else:
            g.ok(f"{method} params present and not disabled")
        if expect_method and expect_method.lower() not in name.lower():
            g.warn(f"expected method '{expect_method}' but summary carries "
                   f"'{method}' params — check you're reading the right run")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="directory containing summary.json + results.jsonl")
    ap.add_argument("--expect-method", metavar="NAME",
                    help="method name that SHOULD have engaged (e.g. dycoke)")
    ap.add_argument("--strict", action="store_true",
                    help="treat WARN as failure too")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"{RED}[FAIL]{RESET} not a directory: {args.run_dir}")
        return 2

    print(f"Checking run: {args.run_dir}\n")
    g = Gate()

    summary = load_summary(args.run_dir, g)
    rows = iter_results(args.run_dir, g)

    if summary is not None:
        check_completeness(summary, rows, g)
        check_na_accounting(summary, g)
        check_accuracy_band(summary, g)
        check_method_engaged(summary, args.expect_method, g)
    check_predictions_nondegenerate(rows, g)

    print()
    n_fail, n_warn = len(g.fails), len(g.warns)
    if n_fail:
        print(f"{RED}RESULT: FAIL{RESET} — {n_fail} failure(s), {n_warn} warning(s). "
              f"Do NOT record this run.")
        return 1
    if n_warn and args.strict:
        print(f"{YELLOW}RESULT: FAIL (strict){RESET} — {n_warn} warning(s).")
        return 1
    if n_warn:
        print(f"{YELLOW}RESULT: PASS with {n_warn} warning(s).{RESET}")
        return 0
    print(f"{GREEN}RESULT: PASS{RESET} — run looks trustworthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

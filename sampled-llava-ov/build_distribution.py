#!/usr/bin/env python3
"""
build_distribution.py — A/B/C/D answer distribution for the sampled LLaVA-OV runs.

Emits a grouped bar chart (PNG) plus a markdown table of the underlying counts,
so the numbers are readable without opening the image.

What this shows: how often each method answered A, B, C or D across the 4,018
scoreable questions, against the ground-truth distribution. A method whose bars
collapse onto one letter is guessing, not reasoning — which an accuracy figure
alone can hide.

These are SAMPLED runs (temperature 0.7, top_p 0.9). They are not comparable to
the gated greedy numbers and must not be merged into master-results.md.

Usage
-----
    # after scripts/fetch_results.sh has cached the runs
    python3 sampled-llava-ov/build_distribution.py --local

    # or on Carya against $HUVLLM_RESULTS
    python3 sampled-llava-ov/build_distribution.py
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("baseline", "samp_baseline_run"), ("DyCoke", "samp_dycoke_run"),
        ("FlashVID", "samp_flashvid_run"), ("HoliTom", "samp_holitom_run"),
        ("MDP3", "samp_mdp3_run"), ("AIM", "samp_aim_run"),
        ("VideoITG", "samp_videoitg_run"), ("STTM", "samp_sttm_run")]
LETTERS = ["A", "B", "C", "D"]


def parse_letter(pred):
    """Same rule the eval scripts score with, so the chart matches the results."""
    if not pred:
        return None
    m = re.search(r"\b([A-D])\b", str(pred).upper())
    if m:
        return m.group(1)
    c = str(pred).strip().upper()[:1]
    return c if c in LETTERS else None


def load(res_root, d):
    p = os.path.join(res_root, d, "results.jsonl")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true",
                    help="read from results-cache/ instead of $HUVLLM_RESULTS")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs"))
    args = ap.parse_args()

    res = (os.path.join(HERE, "..", "results-cache") if args.local
           else os.environ.get("HUVLLM_RESULTS", "/project/rhu/dpalfaro/results"))

    dist, truth, missing = {}, {L: 0 for L in LETTERS}, []
    got_truth = False
    for label, d in RUNS:
        rows = load(res, d)
        if rows is None:
            missing.append(label)
            continue
        counts = {L: 0 for L in LETTERS}
        unparsed = 0
        for r in rows:
            if r.get("correct") is None:      # NA — unanswerable, excluded
                continue
            c = parse_letter(r.get("prediction"))
            if c:
                counts[c] += 1
            else:
                unparsed += 1
            if not got_truth:
                g = str(r.get("ground_truth", "")).strip().upper()[:1]
                if g in LETTERS:
                    truth[g] += 1
        dist[label] = (counts, unparsed)
        got_truth = True

    if not dist:
        print("No sampled runs found yet.", file=sys.stderr)
        print(f"  looked in: {res}", file=sys.stderr)
        print(f"  missing:   {', '.join(m for m, _ in RUNS)}", file=sys.stderr)
        return 1

    # ---- markdown table --------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    md = os.path.join(args.out, "SAMPLED_DISTRIBUTION.md")
    with open(md, "w") as f:
        f.write("# Answer distribution — sampled LLaVA-OV runs\n\n")
        f.write("**These are SAMPLED runs** (`temperature=0.7, top_p=0.9, seed=0`), "
                "not the gated greedy results. They cannot pass the divergence "
                "gate — two sampled runs differ by chance — so they are not "
                "comparable to the numbers in `master-results.md` and must not "
                "be merged into it.\n\n")
        f.write("Counts are over the 4,018 scoreable questions (NA excluded). "
                "*unparsed* = generations with no recoverable A–D letter.\n\n")
        f.write("| Method | A | B | C | D | unparsed | most-picked |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        tot_t = sum(truth.values()) or 1
        f.write("| *ground truth* | "
                + " | ".join(f"*{100.0*truth[L]/tot_t:.1f}%*" for L in LETTERS)
                + " | *—* | *—* |\n")
        for label, _ in RUNS:
            if label not in dist:
                continue
            counts, unp = dist[label]
            tot = sum(counts.values()) or 1
            top = max(counts, key=counts.get)
            skew = 100.0 * counts[top] / tot
            flag = " ⚠️" if skew > 40 else ""
            f.write(f"| **{label}** | "
                    + " | ".join(f"{100.0*counts[L]/tot:.1f}%" for L in LETTERS)
                    + f" | {unp} | {top} {skew:.0f}%{flag} |\n")
        f.write("\n⚠️ = one letter takes >40% of answers, i.e. the model is "
                "leaning on a default rather than discriminating. Ground truth "
                "is near-uniform, so a healthy method should be too.\n")
        if missing:
            f.write(f"\n*Not yet run: {', '.join(missing)}.*\n")
    print(f"wrote {md}")

    # ---- chart -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — markdown table written, chart skipped",
              file=sys.stderr)
        return 0

    labels = [l for l, _ in RUNS if l in dist]
    x = np.arange(len(labels))
    w = 0.2
    # colourblind-safe, distinguishable in greyscale
    colours = ["#4477AA", "#66CCEE", "#228833", "#CCBB44"]
    fig, ax = plt.subplots(figsize=(max(8, 1.35 * len(labels)), 4.6))
    for i, L in enumerate(LETTERS):
        vals = [100.0 * dist[l][0][L] / (sum(dist[l][0].values()) or 1)
                for l in labels]
        ax.bar(x + (i - 1.5) * w, vals, w, label=f"answered {L}", color=colours[i])
    tot_t = sum(truth.values()) or 1
    ax.axhline(100.0 / 4, ls="--", lw=1, color="#BB5566",
               label="uniform (25%)", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("% of scoreable answers")
    ax.set_title("MotionBench answer distribution — LLaVA-OV-7B, sampled "
                 "(temp 0.7, top_p 0.9)")
    ax.legend(ncol=5, fontsize=8, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    png = os.path.join(figdir, "answer_distribution_llava_ov.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

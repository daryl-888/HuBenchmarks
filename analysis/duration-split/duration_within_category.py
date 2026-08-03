#!/usr/bin/env python3
"""
duration_within_category.py -- the duration test, with question type controlled.

The naive short/long split is confounded twice over:

  1. Category mix. At a 3 s cut, Motion-related Objects is 33.7% of the short
     half but 11.7% of the long half, and Repetition Count is 3.7% vs 12.0%.
     Duration is largely a proxy for question type.
  2. Floor effect. The backbone scores 62.5 on short clips and 49.4 on long, so
     a method degenerating toward a fixed answer loses more points wherever the
     baseline was better, with no duration mechanism involved.

This holds question type fixed and asks, inside each category, whether the loss
differs between short and long clips. If duration matters, the gap should
survive; if it was mix and floor, it should vanish.

    python3 analysis/duration-split/duration_within_category.py [--cut 5]
"""
import argparse
import json
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, "results-cache")
DUR = {k: v[0] for k, v in json.load(open(os.path.join(CACHE, "_video_durations.json"))).items()}

BASE = "fastv_run1"
METHODS = [("FastV", "w2_fastv_run"), ("PruneVID-OV", "w2_prunevid_ov_run"),
           ("FlashVID", "w2_flashvid_run"), ("DyCoke", "w2_dycoke_run")]


def load(name):
    out = {}
    for line in open(os.path.join(CACHE, name, "results.jsonl")):
        r = json.loads(line)
        if r.get("ground_truth", "").strip().upper() == "NA" or r.get("correct") is None:
            continue
        out[r["idx"]] = r
    return out


def mcnemar(a, b, keys):
    br = sum(1 for k in keys if a[k]["correct"] == 1 and b[k]["correct"] == 0)
    fx = sum(1 for k in keys if a[k]["correct"] == 0 and b[k]["correct"] == 1)
    n = br + fx
    return br, fx, ((abs(br - fx) - 1) ** 2 / n if n else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=float, default=5.0)
    args = ap.parse_args()
    cut = args.cut

    base = load(BASE)
    bycat = defaultdict(lambda: ([], []))
    for k, r in base.items():
        d = DUR.get(r["video_path"])
        if d is None:
            continue
        bycat[r["question_type"]][0 if d < cut else 1].append(k)

    for label, name in METHODS:
        try:
            run = load(name)
        except FileNotFoundError:
            continue
        print("\n" + "=" * 96)
        print("%s vs backbone, within category, duration cut %.0f s" % (label, cut))
        print("=" * 96)
        print("  %-26s %20s %20s %10s" % ("category", "SHORT <%.0fs" % cut, "LONG >=%.0fs" % cut, ""))
        print("  %-26s %6s %6s %7s %6s %6s %7s %10s"
              % ("", "n", "base", "delta", "n", "base", "delta", "dS-dL"))
        tot_s = tot_l = 0
        wsum = 0.0
        for cat in sorted(bycat):
            ks, kl = bycat[cat]
            ks = [k for k in ks if k in run]
            kl = [k for k in kl if k in run]
            if len(ks) < 30 or len(kl) < 30:
                note = "(n too small)"
                print("  %-26s %6d %6s %7s %6d %6s %7s %10s"
                      % (cat, len(ks), "-", "-", len(kl), "-", "-", note))
                continue
            bs = 100.0 * sum(base[k]["correct"] for k in ks) / len(ks)
            bl = 100.0 * sum(base[k]["correct"] for k in kl) / len(kl)
            ms = 100.0 * sum(run[k]["correct"] for k in ks) / len(ks)
            ml = 100.0 * sum(run[k]["correct"] for k in kl) / len(kl)
            ds, dl = ms - bs, ml - bl
            n = len(ks) + len(kl)
            wsum += (ds - dl) * n
            tot_s += len(ks)
            tot_l += len(kl)
            print("  %-26s %6d %6.1f %+7.2f %6d %6.1f %+7.2f %+10.2f"
                  % (cat, len(ks), bs, ds, len(kl), bl, dl, ds - dl))
        if tot_s + tot_l:
            print("  %-26s %6s %6s %7s %6s %6s %7s %+10.2f"
                  % ("WEIGHTED MEAN", "", "", "", "", "", "", wsum / (tot_s + tot_l)))


if __name__ == "__main__":
    main()

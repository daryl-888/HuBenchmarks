#!/usr/bin/env python3
"""
collapse_by_duration.py -- is FastV's collapse worse on short clips, or does it
only look that way?

split_by_duration.py finds a significant interaction at a 3 s cut: FastV loses
19.8 points on clips under 3 s versus 14.6 above, and PruneVID-OV behaves the
same. That is confounded. The backbone is much stronger on short clips (62.5 vs
49.4), so ANY method that degenerates toward a fixed answering strategy will
show a larger drop wherever the baseline was better, with no duration mechanism
involved.

Two measures that are not sensitive to baseline skill:

  1. D-rate -- share of answers on option D. The collapse signature is ~47%
     against a 23.6% ground-truth rate. If short clips collapse harder, their
     D-rate must be higher. A pure floor effect leaves it flat.
  2. Category mix -- short and long halves must not be answering different
     questions, or the whole split measures question type, not duration.

    python3 analysis/duration-split/collapse_by_duration.py
"""
import json
import os
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, "results-cache")
DUR = json.load(open(os.path.join(CACHE, "_video_durations.json")))
DUR = {k: v[0] for k, v in DUR.items()}

LETTER = ("A", "B", "C", "D")


def rows(name):
    p = os.path.join(CACHE, name, "results.jsonl")
    out = []
    for line in open(p):
        r = json.loads(line)
        if r.get("ground_truth", "").strip().upper() == "NA" or r.get("correct") is None:
            continue
        out.append(r)
    return out


def letter_of(pred):
    import re
    m = re.search(r"\b([A-D])\b", (pred or "").upper())
    return m.group(1) if m else None


def dist(rs):
    c = Counter(letter_of(r["prediction"]) for r in rs)
    n = len(rs)
    return [100.0 * c[l] / n for l in LETTER], n


RUNS = [
    ("LLaVA-OV baseline", "fastv_run1"),
    ("FastV keep10", "s1_fastv_r10_run"),
    ("FastV keep15", "w2_fastv_run"),
    ("FastV keep25", "s1_fastv_r25_run"),
    ("PruneVID-OV k15", "w2_prunevid_ov_run"),
    ("FlashVID keep15", "w2_flashvid_run"),
    ("DyCoke keep15", "w2_dycoke_run"),
]

for cut in (3.0, 5.0):
    print("\n" + "=" * 92)
    print("ANSWER DISTRIBUTION BY DURATION, cut at %.0f s" % cut)
    print("=" * 92)
    print("%-20s %26s %26s %10s" % ("run", "SHORT <%.0fs" % cut, "LONG >=%.0fs" % cut, "D short"))
    print("%-20s %26s %26s %10s" % ("", "A     B     C     D", "A     B     C     D", "- D long"))
    for label, name in RUNS:
        try:
            rs = rows(name)
        except FileNotFoundError:
            print("%-20s (not cached)" % label)
            continue
        s = [r for r in rs if DUR.get(r["video_path"]) is not None and DUR[r["video_path"]] < cut]
        l = [r for r in rs if DUR.get(r["video_path"]) is not None and DUR[r["video_path"]] >= cut]
        ds, _ = dist(s)
        dl, _ = dist(l)
        print("%-20s  %5.1f %5.1f %5.1f %5.1f      %5.1f %5.1f %5.1f %5.1f     %+6.1f"
              % (label, ds[0], ds[1], ds[2], ds[3], dl[0], dl[1], dl[2], dl[3], ds[3] - dl[3]))

print("\n" + "=" * 92)
print("CATEGORY MIX -- are the halves asking the same questions?")
print("=" * 92)
base = rows("fastv_run1")
for cut in (3.0, 5.0):
    s = [r for r in base if DUR.get(r["video_path"]) is not None and DUR[r["video_path"]] < cut]
    l = [r for r in base if DUR.get(r["video_path"]) is not None and DUR[r["video_path"]] >= cut]
    cs, cl = Counter(r["question_type"] for r in s), Counter(r["question_type"] for r in l)
    print("\ncut %.0fs   short n=%d   long n=%d" % (cut, len(s), len(l)))
    print("  %-26s %9s %9s %8s" % ("category", "short %", "long %", "diff"))
    for k in sorted(set(cs) | set(cl)):
        ps, pl = 100.0 * cs[k] / len(s), 100.0 * cl[k] / len(l)
        print("  %-26s %8.1f %9.1f %+8.1f" % (k, ps, pl, ps - pl))

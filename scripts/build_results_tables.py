#!/usr/bin/env python3
"""
build_results_tables.py — generate every table in docs/RESULTS.md from the
cached per-sample predictions, so the document cannot drift from the data.

Emits, in order:
  1. Main table      — every method in its ORIGINAL published configuration
  2. LLaVA-OV-7B     — all methods on that backbone, 15% where applicable
  3. Qwen3-VL-8B     — all methods on that backbone, 15% where applicable
  4. Retention sweep — 0.10 / 0.15 / 0.25, ONLY methods intact at 0.15
  5. Native backbones — methods bound to a backbone of their own

Every section carries: name, venue, year, overall, Δ vs that backbone's
baseline, and the six MotionBench subcategories.

Usage:  python3 scripts/build_results_tables.py            # writes docs/RESULTS.md
        python3 scripts/build_results_tables.py --stdout   # print instead
"""
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.environ.get("HUVLLM_CACHE", os.path.join(ROOT, "results-cache"))

CATS = ["Action Order", "Camera Motion", "Location-related Motion",
        "Motion Recognition", "Motion-related Objects", "Repetition Count"]
ABBR = ["AO", "CM", "LM", "MR", "MO", "RC"]

BASE_OV = "fastv_run1"              # bare LLaVA-OV backbone (old FastV stub)
BASE_QW = "qwen3vl_baseline_run1"

# name -> (venue, year). Verified against each method's live GitHub repo,
# July 2026 -- see docs/UPSTREAM_CROSSREF.md. Eight of these were previously
# wrong; MDP3 in particular had been credited with an ICCV acceptance it does
# not have. Do not edit without a source.
META = {
    "DyCoke":    ("CVPR", 2025),        # arXiv 2411.15024 (NOT 2411.14401 = DyTo)
    "FlashVID":  ("ICLR (Oral)", 2026),
    "HoliTom":   ("NeurIPS", 2025),
    "MDP3":      ("arXiv", 2025),       # 2501.02885 -- preprint, no venue
    "AIM":       ("ICCV", 2025),
    "VideoITG":  ("CVPR (Highlight)", 2026),
    "STTM":      ("ICCV", 2025),
    "FastV":     ("ECCV (Oral)", 2024),
    "PruneVID":  ("ACL", 2025),
    "VisionZip": ("CVPR", 2025),
    "DyTo":      ("ICCV", 2025),        # arXiv 2411.14401
}

# Methods intact at 15% retention AND actually swept at 0.10/0.15/0.25.
# FastV collapses; PruneVID has no 0.15 measurement (its middle cell is 0.50).
# See docs/RETENTION_DIAGNOSIS.md.
SWEEP = [
    ("FlashVID", "s1_flashvid_r10_run", "w2_flashvid_run", "s1_flashvid_r25_run", "LLaVA-OV"),
    ("HoliTom",  "s1_holitom_r10_run",  "w2_holitom_run",  "s1_holitom_r25_run",  "LLaVA-OV"),
    ("FlashVID", "s3_flashvid_r10_run", "w3_flashvid_run", "s3_flashvid_r25_run", "Qwen3-VL"),
    ("HoliTom",  "s3_holitom_r10_run",  "w3_holitom_run",  "s3_holitom_r25_run",  "Qwen3-VL"),
]

# A trailing "!" marks a cell the upstream cross-reference flags: a port to a
# backbone the authors do not support, or a configuration matching no published
# setting. See docs/UPSTREAM_CROSSREF.md.
STAGE1 = [("DyCoke", "w2_dycoke_run"), ("FlashVID", "w2_flashvid_run"),
          ("HoliTom", "w2_holitom_run"), ("MDP3", "w2_mdp3_run"),
          ("VideoITG!", "w2_videoitg_run"), ("AIM", "w2_aim_run"),
          ("STTM", "w2_sttm_run"), ("PruneVID!", "w2_prunevid_ov_run"),
          ("FastV", "w2_fastv_run")]

STAGE3 = [("PruneVID", "w3_prunevid_run"), ("DyCoke", "w3_dycoke_run"),
          ("HoliTom", "w3_holitom_run"), ("MDP3", "w3_mdp3_run"),
          ("FastV", "w3_fastv_run"), ("VisionZip!", "w3_visionzip_run"),
          ("STTM!", "w3_sttm_run"), ("FlashVID!", "w3_flashvid_run"),
          ("VideoITG", "w3_videoitg_run"), ("AIM!", "w3_aim_run")]

NATIVE = [("STTM", "sttm_llavavid_t80_full", "LLaVA-Video-7B", "thresh=0.80 temporal=0.65 root=1"),
          ("PruneVID", "w2_prunevid_run", "PLLaVA-7B", "cluster=0.50 seg=0.25 layer=10 alpha=0.4 tau=0.8"),
          ("DyTo", "ob_dyto_run", "LLaVA-NeXT Vicuna-7B", "reconstructed TW-FINCH"),
          ("VisionZip", "w2_visionzip_run", "LLaVA-1.5-7B", "8 frames, dominant=54 contextual=10")]

# Main table: each method at its ORIGINAL published configuration.
# run, backbone, the setting that makes it "original", and any caveat.
ORIGINAL = [
    ("HoliTom",  "w2_holitom_run",      "LLaVA-OV-7B",     "RETAIN=0.15 T=0.80 k=18 r=0.5", ""),
    ("DyCoke",   "w2_dycoke_run",       "LLaVA-OV-7B",     "l=3 p=0.7 k=0.7", ""),
    ("FlashVID", "s1_flashvid_r25_run", "LLaVA-OV-7B",     "retention=0.25", ""),
    ("MDP3",     "w2_mdp3_run",         "LLaVA-OV-7B",     "pool=32 select=8", ""),
    ("VideoITG", "w2_videoitg_run",     "LLaVA-OV-7B",     "512 sampled / 32 selected", ""),
    ("AIM",      "w2_aim_run",          "LLaVA-OV-7B",     "4-step bipartite merge + PageRank", ""),
    # Bound to a backbone of their own -- these are the published pairings.
    ("STTM",     "sttm_llavavid_t80_full", "LLaVA-Video-7B", "thresh=0.80 temporal=0.65 root=1", "own backbone; no matched baseline"),
    ("PruneVID", "w2_prunevid_run",     "PLLaVA-7B",       "cluster=0.50 seg=0.25 alpha=0.4 tau=0.8", "own backbone; no matched baseline"),
    ("DyTo",     "ob_dyto_run",         "Vicuna-7B",       "spatial_tome_finch_dynamic", "own backbone; no matched baseline; reconstructed"),
    ("VisionZip", "w2_visionzip_run",   "LLaVA-1.5-7B",    "dominant=54 contextual=10, 8f", "own backbone; no matched baseline"),
    ("FastV",    "w2_fastv_run",        "LLaVA-OV-7B",     "keep 15%", "NOT a published setting \u2014 FastV released no video config, so this is our standardized 15%, shown for continuity with \u00a72"),
]


def load(run):
    p = os.path.join(CACHE, run, "results.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def acc(rows):
    s = [r for r in rows if r.get("correct") is not None]
    return 100.0 * sum(1 for r in s if r["correct"]) / len(s), len(s)


def per_cat(rows):
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("correct") is None:
            continue
        d[r.get("question_type")][1] += 1
        if r["correct"]:
            d[r.get("question_type")][0] += 1
    return {c: (100.0 * d[c][0] / d[c][1] if d[c][1] else float("nan")) for c in CATS}


def r1(x):
    """Round half away from zero to 1 dp — matches the paper's convention."""
    return f"{x + (1e-9 if x >= 0 else -1e-9):.1f}"


def r2(x):
    return f"{x + (1e-9 if x >= 0 else -1e-9):.2f}"


def delta(x):
    return ("+" if x >= 0 else "−") + r2(abs(x))


def mcnemar(base_rows, m_rows):
    """chi2 on discordant pairs; returns (chi2, significant)."""
    B = {r["idx"]: r for r in base_rows if r.get("correct") is not None}
    M = {r["idx"]: r for r in m_rows if r.get("correct") is not None}
    sh = set(B) & set(M)
    br = sum(1 for i in sh if B[i]["correct"] and not M[i]["correct"])
    fx = sum(1 for i in sh if not B[i]["correct"] and M[i]["correct"])
    chi = ((abs(br - fx) - 1) ** 2 / (br + fx)) if (br + fx) else 0.0
    return chi, chi >= 3.84


def catcells(rows):
    pc = per_cat(rows)
    return " | ".join(r1(pc[c]) for c in CATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    bov = load(BASE_OV)
    bqw = load(BASE_QW)
    if bov is None or bqw is None:
        print("missing a baseline run in results-cache/", file=sys.stderr)
        return 1
    aov, n_ov = acc(bov)
    aqw, _ = acc(bqw)
    base = {"LLaVA-OV": aov, "LLaVA-OV-7B": aov,
            "Qwen3-VL": aqw, "Qwen3-VL-8B": aqw}

    O = []
    w = O.append
    w("# Results\n")
    w(f"MotionBench: 8,052 questions, **{n_ov:,} scoreable** (4,034 `NA` excluded) — the")
    w("denominator for every accuracy below. Greedy decoding, 32 frames. Categories:")
    w("**AO** Action Order · **CM** Camera Motion · **LM** Location-related Motion ·")
    w("**MR** Motion Recognition · **MO** Motion-related Objects · **RC** Repetition Count.")
    w("Chance is 25%.\n")
    w("Differences below **±1.54 pt** are not resolvable at this n and are not ranked.")
    w("`*` marks significance by McNemar on paired predictions (χ² ≥ 3.84, p<0.05).")
    w("Every figure is computed from per-sample predictions by")
    w("[`scripts/build_results_tables.py`](../scripts/build_results_tables.py) — do not hand-edit.\n")
    w("| Backbone | Baseline | AO | CM | LM | MR | MO | RC |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    w(f"| LLaVA-OV-7B | {r2(aov)}% | {catcells(bov)} |")
    w(f"| Qwen3-VL-8B | {r2(aqw)}% | {catcells(bqw)} |")
    w("")

    # ---- 1. main: original configurations --------------------------------
    w("## 1. Every method in its original configuration\n")
    w("Each method at the setting its authors published, on the backbone it was")
    w("designed for where one exists. Methods whose published setting is 15%")
    w("retention appear at 15%; others at their own default, stated per row.\n")
    w("| Method | Venue | Year | Backbone | Setting | Overall | Δ base | AO | CM | LM | MR | MO | RC |")
    w("|---|:--:|:--:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    SUP = "¹²³⁴⁵⁶⁷⁸⁹"
    rows_sorted = []
    for name, run, bb, setting, note in ORIGINAL:
        rr = load(run)
        if rr is None:
            continue
        a, _ = acc(rr)
        rows_sorted.append((a, name, rr, bb, setting, note))
    notes = []
    for a, name, rr, bb, setting, note in sorted(rows_sorted, reverse=True):
        ven, yr = META.get(name, ("—", "—"))
        if bb in base:
            _, sig = mcnemar(bov if bb.startswith("LLaVA-OV") else bqw, rr)
            d = delta(a - base[bb]) + ("*" if sig else "")
        else:
            d = "—"
        mark = ""
        if note:
            notes.append((SUP[len(notes)], name, note))
            mark = notes[-1][0]
        w(f"| **{name}**{mark} | {ven} | {yr} | {bb} | {setting} | **{r2(a)}%** | {d} | {catcells(rr)} |")
    w("")
    for sup, name, note in notes:
        w(f"{sup} **{name}** — {note}.  ")
    w("")

    # ---- 2 & 3. per-backbone ---------------------------------------------
    for title, stage, bkey, bl in (("2. LLaVA-OV-7B", STAGE1, "LLaVA-OV", bov),
                                   ("3. Qwen3-VL-8B", STAGE3, "Qwen3-VL", bqw)):
        w(f"## {title}\n")
        w("All methods on this backbone at 15% nominal retention where the method")
        w("exposes one; frame-selection methods keep their defaults.\n")
        w("| Method | Venue | Year | Overall | Δ base | AO | CM | LM | MR | MO | RC |")
        w("|---|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|")
        got = []
        for name, run in stage:
            rr = load(run)
            if rr is None:
                continue
            a, _ = acc(rr)
            got.append((a, name, rr))
        placed = False
        for a, name, rr in sorted(got, reverse=True):
            if not placed and a < base[bkey]:
                w(f"| *backbone* | — | — | *{r2(base[bkey])}%* | — | {catcells(bl)} |")
                placed = True
            flag = name.endswith("!")
            key = name.rstrip("!")
            ven, yr = META.get(key, ("—", "—"))
            _, sig = mcnemar(bl, rr)
            disp = key + ("<sup>!</sup>" if flag else "")
            w(f"| {disp} | {ven} | {yr} | **{r2(a)}%** | "
              f"{delta(a - base[bkey])}{'*' if sig else ''} | {catcells(rr)} |")
        if not placed:
            w(f"| *backbone* | — | — | *{r2(base[bkey])}%* | — | {catcells(bl)} |")
        w("")
        w("<sup>!</sup> flagged by the upstream cross-reference — a port to a backbone")
        w("the authors do not support, or a configuration matching no published")
        w("setting. Not a method result as published; see")
        w("[UPSTREAM_CROSSREF.md](UPSTREAM_CROSSREF.md).\n")

    # ---- 4. retention sweep ----------------------------------------------
    w("## 4. Retention sweep — 0.10 / 0.15 / 0.25\n")
    w("**Only methods that remain intact at 0.15 and were actually run at all three**")
    w("**ratios appear here.** FastV is excluded: it collapses across this entire band")
    w("(36.06 / 36.78 / 36.73 on LLaVA-OV) and its numbers describe a failure mode, not")
    w("a retention response. PruneVID is excluded: it has no 0.15 measurement — the cell")
    w("previously labelled 0.15 was its published `cluster_ratio=0.50`. Both are")
    w("diagnosed in [RETENTION_DIAGNOSIS.md](RETENTION_DIAGNOSIS.md).\n")
    for bb in ("LLaVA-OV", "Qwen3-VL"):
        w(f"### {bb}-{'7B' if bb == 'LLaVA-OV' else '8B'} (baseline {r2(base[bb])}%)\n")
        w("| Method | Venue | Year | r | Overall | Δ base | AO | CM | LM | MR | MO | RC |")
        w("|---|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, r10, r15, r25, who in SWEEP:
            if who != bb:
                continue
            ven, yr = META.get(name, ("—", "—"))
            for lbl, run in (("0.10", r10), ("0.15", r15), ("0.25", r25)):
                rr = load(run)
                if rr is None:
                    continue
                a, _ = acc(rr)
                nm = f"**{name}**" if lbl == "0.10" else ""
                v = ven if lbl == "0.10" else ""
                y = yr if lbl == "0.10" else ""
                _, sig = mcnemar(bov if bb == "LLaVA-OV" else bqw, rr)
                w(f"| {nm} | {v} | {y} | {lbl} | {r2(a)}% | "
                  f"{delta(a - base[bb])}{'*' if sig else ''} | {catcells(rr)} |")
        w("")

    # ---- 5. native backbones ---------------------------------------------
    w("## 5. Methods on their own backbones\n")
    w("Bound to a backbone neither standardized model can host. **No Δ is given:**")
    w("no matched baseline exists, so these are absolute scores and are not")
    w("comparable to the tables above.\n")
    w("| Method | Venue | Year | Backbone | Setting | Overall | AO | CM | LM | MR | MO | RC |")
    w("|---|:--:|:--:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, run, bb, setting in NATIVE:
        rr = load(run)
        if rr is None:
            continue
        a, _ = acc(rr)
        ven, yr = META.get(name, ("—", "—"))
        w(f"| {name} | {ven} | {yr} | {bb} | {setting} | **{r2(a)}%** | {catcells(rr)} |")
    w("")
    w("VisionZip patches `CLIPVisionTower`, which LLaVA-OV does not have, so its")
    w("complete form runs only on LLaVA-1.5-7B and at 8 frames. DyTo is not")
    w("reproducible from its published artifacts and is reported as")
    w("*DyTo (reconstructed TW-FINCH)*; it is also the one number here with")
    w("execution evidence but no divergence check, since no baseline exists on its")
    w("backbone. See [UPSTREAM_DEFECTS.md](UPSTREAM_DEFECTS.md).\n")
    w("---\n")
    w("Related: [RETENTION_DIAGNOSIS.md](RETENTION_DIAGNOSIS.md) ·")
    w("[FASTV_COLLAPSE_ANALYSIS.md](FASTV_COLLAPSE_ANALYSIS.md) ·")
    w("[METHODOLOGY.md](METHODOLOGY.md) · [DETERMINISM_AND_VALIDITY.md](DETERMINISM_AND_VALIDITY.md)")

    out = "\n".join(O) + "\n"
    if args.stdout:
        sys.stdout.write(out)
    else:
        p = os.path.join(ROOT, "docs", "RESULTS.md")
        open(p, "w").write(out)
        print(f"wrote {p}  ({len(O)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

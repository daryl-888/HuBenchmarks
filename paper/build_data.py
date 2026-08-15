#!/usr/bin/env python3
"""Single source of truth for the revised paper.

Computes, per run, from results-cache/*/results.jsonl:
  - overall accuracy (scoreable n=4018)
  - per-category accuracy (6 MotionBench categories)
  - delta vs the matching backbone baseline, overall and per category
  - McNemar chi2 (for significance MARKING only, not for a table column)
  - divergence over the full 8,052 rows (project convention) and over scoreable only

Emits data.json consumed by make_figures.py and gen_tables.py.
"""
import json, os
from collections import defaultdict

# Repo-relative so this runs anywhere the repo is checked out. Override with
# $HUVLLM_CACHE / $HUVLLM_DATA if the cache lives elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CACHE = os.environ.get("HUVLLM_CACHE", os.path.join(_ROOT, "results-cache"))
OUT = os.environ.get("HUVLLM_DATA", os.path.join(_HERE, "data.json"))

CATS = ["Action Order", "Camera Motion", "Location-related Motion",
        "Motion Recognition", "Motion-related Objects", "Repetition Count"]
CAT_ABBR = {"Action Order": "AO", "Camera Motion": "CM",
            "Location-related Motion": "LM", "Motion Recognition": "MR",
            "Motion-related Objects": "MO", "Repetition Count": "RC"}


def load(run):
    rows = []
    with open(os.path.join(CACHE, run, "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    with open(os.path.join(CACHE, run, "summary.json")) as f:
        summ = json.load(f)
    return rows, summ


def overall(rows):
    s = [r for r in rows if r.get("correct") is not None]
    c = sum(1 for r in s if r["correct"])
    return c / len(s) * 100, c, len(s)


def per_cat(rows):
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("correct") is None:
            continue
        k = r.get("question_type", "Unknown")
        d[k][1] += 1
        if r["correct"]:
            d[k][0] += 1
    return {k: (v[0] / v[1] * 100, v[1]) for k, v in d.items()}


def compare(base_rows, m_rows):
    """Divergence + McNemar. Divergence reported on BOTH denominators because the
    project's docs use all 8,052 rows while accuracy uses the 4,018 scoreable."""
    b_all = {r["idx"]: r for r in base_rows}
    m_all = {r["idx"]: r for r in m_rows}
    shared_all = set(b_all) & set(m_all)
    differ_all = sum(1 for i in shared_all
                     if b_all[i].get("prediction") != m_all[i].get("prediction"))

    b = {r["idx"]: r for r in base_rows if r.get("correct") is not None}
    m = {r["idx"]: r for r in m_rows if r.get("correct") is not None}
    shared = set(b) & set(m)
    differ_score = sum(1 for i in shared if b[i]["prediction"] != m[i]["prediction"])
    broke = sum(1 for i in shared if b[i]["correct"] and not m[i]["correct"])
    fixed = sum(1 for i in shared if not b[i]["correct"] and m[i]["correct"])
    nd = broke + fixed
    chi2 = ((abs(broke - fixed) - 1) ** 2 / nd) if nd else 0.0
    return {
        "differ_all8052": differ_all, "n_all": len(shared_all),
        "differ_scoreable": differ_score,
        "broke": broke, "fixed": fixed,
        "chi2": round(chi2, 2), "significant": chi2 >= 3.84,
    }


BASE_OV = "fastv_run1"          # bare LLaVA-OV backbone (the old FastV stub)
BASE_QW = "qwen3vl_baseline_run1"

STAGE1 = [("DyCoke", "w2_dycoke_run"), ("FlashVID", "w2_flashvid_run"),
          ("HoliTom", "w2_holitom_run"), ("MDP3", "w2_mdp3_run"),
          ("VideoITG", "w2_videoitg_run"), ("AIM", "w2_aim_run"),
          ("STTM", "w2_sttm_run"),
          # w2_fastv_run predates the PrunableDynamicCache fix (see docs/notes):
          # the cache round-trip discarded FastV's pruning entirely on every prior
          # run. capfix_fastv_r15_run is the same 15% point re-run with the fix.
          # It reproduces w2_fastv_run's accuracy almost exactly (36.78% both) --
          # confirms the LLaVA-OV collapse is real model behavior, not the bug.
          ("FastV", "capfix_fastv_r15_run"),
          ("VisionZip (contextual)", "visionzip_ov_r15_run"),
          ("PruneVID-OV", "w2_prunevid_ov_run")]

STAGE3 = [("PruneVID", "w3_prunevid_run"), ("DyCoke", "w3_dycoke_run"),
          ("HoliTom", "w3_holitom_run"), ("MDP3", "w3_mdp3_run"),
          ("FastV", "w3_fastv_run"), ("VisionZip (contextual)", "w3_visionzip_run"),
          ("STTM", "s3_sttm_pub_run"), ("FlashVID", "s3_flashvid_official_run"),
          ("VideoITG", "w3_videoitg_run"), ("AIM", "w3_aim_run")]

# PruneVID's authors publish exactly one cluster_ratio (0.5); it was never run
# at 0.15 on either backbone. w2_prunevid_ov_run / w3_prunevid_run ARE the
# cluster_ratio=0.5 runs -- labeling them "0.15" in a retention sweep is
# wrong. Keyed by true ratio (0.10 / 0.25 / 0.50) here; callers must not
# treat PruneVID's "0.50" slot as aligned with the other methods' "0.25" column.
RETENTION = {
    # All six points use the PrunableDynamicCache-fixed implementation
    # (capfix_fastv_r*_run) -- physical-prune + no-legacy-cache-roundtrip, so
    # the pruning actually survives instead of being silently discarded.
    # Every point reproduces its pre-fix counterpart almost exactly (10%:
    # 36.06->36.06, 15%: 36.78->36.78, 20%: 36.93->36.93, 25%: 36.73->36.73,
    # 50%: 36.34->36.31, 75%: 35.69->35.69) -- the collapse is real.
    ("LLaVA-OV", "FastV"): {0.10: "capfix_fastv_r10_run", 0.15: "capfix_fastv_r15_run", 0.20: "capfix_fastv_r20_run", 0.25: "capfix_fastv_r25_run", 0.50: "capfix_fastv_r50_run", 0.75: "capfix_fastv_r75_run"},
    ("LLaVA-OV", "VisionZip"): {0.10: "visionzip_ov_r10_run", 0.15: "visionzip_ov_r15_run", 0.20: "visionzip_ov_r20_run", 0.25: "visionzip_ov_r25_run"},
    ("LLaVA-OV", "FlashVID"): {0.10: "s1_flashvid_r10_run", 0.15: "w2_flashvid_run", 0.20: "s1_flashvid_r20_run", 0.25: "s1_flashvid_r25_run"},
    ("LLaVA-OV", "HoliTom"): {0.10: "s1_holitom_r10_run", 0.15: "w2_holitom_run", 0.20: "s1_holitom_r20_run", 0.25: "s1_holitom_r25_run"},
    ("LLaVA-OV", "PruneVID"): {0.10: "s1_prunevid_r10_run", 0.25: "s1_prunevid_r25_run", 0.50: "w2_prunevid_ov_run"},
    ("Qwen3-VL", "FastV"): {0.10: "s3_fastv_r10_run", 0.15: "w3_fastv_run", 0.20: "s3_fastv_r20_run", 0.25: "s3_fastv_r25_run", 0.50: "s3_fastv_r50_run", 0.75: "s3_fastv_r75_run"},
    # All four points now use the authors'-code implementation (same as the
    # verified 15% cell) -- the old incomplete reimplementation's 10%/25%
    # points (s3_flashvid_r10_run / s3_flashvid_r25_run) are superseded and
    # no longer referenced anywhere.
    ("Qwen3-VL", "FlashVID"): {0.10: "s3_flashvid_official_r10_run", 0.15: "s3_flashvid_official_run", 0.20: "s3_flashvid_official_r20_run", 0.25: "s3_flashvid_official_r25_run"},
    ("Qwen3-VL", "HoliTom"): {0.10: "s3_holitom_r10_run", 0.15: "w3_holitom_run", 0.20: "s3_holitom_r20_run", 0.25: "s3_holitom_r25_run"},
    ("Qwen3-VL", "PruneVID"): {0.10: "s3_prunevid_r10_run", 0.25: "s3_prunevid_r25_run", 0.50: "w3_prunevid_run"},
}

base_ov_rows, base_ov_summ = load(BASE_OV)
base_qw_rows, base_qw_summ = load(BASE_QW)

data = {"baselines": {}, "stage1": {}, "stage3": {}, "retention": {},
        "other": {}, "answer_dist": {}, "meta": {}}


def pack(rows, base_rows=None, base_cats=None):
    acc, corr, n = overall(rows)
    pc = per_cat(rows)
    entry = {
        "overall": round(acc, 2), "correct": corr, "n": n,
        "cats": {CAT_ABBR[c]: round(pc[c][0], 1) for c in CATS if c in pc},
        "cat_n": {CAT_ABBR[c]: pc[c][1] for c in CATS if c in pc},
    }
    if base_cats is not None:
        entry["cat_delta"] = {CAT_ABBR[c]: round(pc[c][0] - base_cats[c][0], 1)
                              for c in CATS if c in pc}
    if base_rows is not None:
        cmp = compare(base_rows, rows)
        entry.update(cmp)
        b_acc, _, _ = overall(base_rows)
        entry["delta"] = round(acc - b_acc, 2)
    return entry


base_ov_cats = per_cat(base_ov_rows)
base_qw_cats = per_cat(base_qw_rows)

data["baselines"]["LLaVA-OV-7B"] = pack(base_ov_rows)
data["baselines"]["Qwen3-VL-8B"] = pack(base_qw_rows)
# backbone-vs-backbone comparison (paired: same questions)
bb = compare(base_ov_rows, base_qw_rows)
bb["delta"] = round(overall(base_qw_rows)[0] - overall(base_ov_rows)[0], 2)
bb["cat_delta"] = {CAT_ABBR[c]: round(base_qw_cats[c][0] - base_ov_cats[c][0], 1) for c in CATS}
data["baselines"]["backbone_gap"] = bb

for name, run in STAGE1:
    rows, _ = load(run)
    data["stage1"][name] = pack(rows, base_ov_rows, base_ov_cats)

for name, run in STAGE3:
    rows, _ = load(run)
    data["stage3"][name] = pack(rows, base_qw_rows, base_qw_cats)

for (backbone, method), levels in RETENTION.items():
    base_rows = base_ov_rows if backbone == "LLaVA-OV" else base_qw_rows
    base_cats = base_ov_cats if backbone == "LLaVA-OV" else base_qw_cats
    for r, run in levels.items():
        rows, _ = load(run)
        data["retention"][f"{backbone}|{method}|{r:.2f}"] = pack(rows, base_rows, base_cats)

# Non-standard backbones present in the cache
for label, run in [("VisionZip (LLaVA-1.5-7B, 8 frames)", "w2_visionzip_run"),
                   ("DyTo (reconstructed TW-FINCH, Vicuna-7B)", "ob_dyto_run")]:
    rows, _ = load(run)
    data["other"][label] = pack(rows)

# Answer-letter distribution, LLaVA-OV runs (diagnostic for the collapse)
for label, run in [("baseline", BASE_OV), ("DyCoke", "w2_dycoke_run"),
                   ("FlashVID", "w2_flashvid_run"), ("HoliTom", "w2_holitom_run"),
                   ("PruneVID-OV", "w2_prunevid_ov_run"), ("FastV", "w2_fastv_run")]:
    rows, _ = load(run)
    s = [r for r in rows if r.get("correct") is not None]
    d = defaultdict(int)
    for r in s:
        p = (r.get("prediction") or "").strip().upper()[:1]
        d[p if p in "ABCD" else "other"] += 1
    data["answer_dist"][label] = {k: round(v / len(s) * 100, 1) for k, v in d.items()}

# ground-truth letter distribution, computed from the data itself
gt = defaultdict(int)
s = [r for r in base_ov_rows if r.get("correct") is not None]
for r in s:
    g = (r.get("ground_truth") or "").strip().upper()[:1]
    gt[g] += 1
data["answer_dist"]["ground truth"] = {k: round(v / len(s) * 100, 1) for k, v in gt.items()}

data["meta"]["cat_n"] = {CAT_ABBR[c]: base_ov_cats[c][1] for c in CATS}
data["meta"]["cat_full"] = {CAT_ABBR[c]: c for c in CATS}

with open(OUT, "w") as f:
    json.dump(data, f, indent=2)

print("BASELINES")
for k in ["LLaVA-OV-7B", "Qwen3-VL-8B"]:
    e = data["baselines"][k]
    print(f"  {k:14s} {e['overall']:.2f}%  ({e['correct']}/{e['n']})  cats={e['cats']}")
print(f"  gap: {bb['delta']:+.2f} pt, chi2={bb['chi2']}, per-cat {bb['cat_delta']}")

print("\nSTAGE 1 (LLaVA-OV)")
for name, _ in STAGE1:
    e = data["stage1"][name]
    print(f"  {name:14s} {e['overall']:6.2f}%  d={e['delta']:+6.2f}  chi2={e['chi2']:7.2f} sig={e['significant']}"
          f"  differ8052={e['differ_all8052']:5d}  cats={e['cats']}")

print("\nSTAGE 3 (Qwen3-VL)")
for name, _ in STAGE3:
    e = data["stage3"][name]
    print(f"  {name:22s} {e['overall']:6.2f}%  d={e['delta']:+6.2f}  chi2={e['chi2']:7.2f} sig={e['significant']}"
          f"  differ8052={e['differ_all8052']:5d}")

print("\nRETENTION")
for k in sorted(data["retention"]):
    e = data["retention"][k]
    print(f"  {k:28s} {e['overall']:6.2f}%  d={e['delta']:+6.2f}  sig={e['significant']}")

print("\nOTHER BACKBONES")
for k, e in data["other"].items():
    print(f"  {k:44s} {e['overall']:6.2f}%  cats={e['cats']}")

print("\nANSWER DIST (LLaVA-OV)")
for k, v in data["answer_dist"].items():
    print(f"  {k:14s} {v}")

print("\nCATEGORY n:", data["meta"]["cat_n"])

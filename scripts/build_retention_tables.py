#!/usr/bin/env python3
"""
build_retention_tables.py — emit the retention-sweep results as markdown.

Produces six tables: {LLaVA-OV-7B, Qwen3-VL-8B} x {0.10, 0.15, 0.25}, plus a
cross-retention summary and the other-backbones table (which has no retention
axis — each of those methods runs on its own native model at its own setting).

Only four methods expose a retention knob on both backbones: FastV, FlashVID,
HoliTom, PruneVID. Methods without one (DyCoke, AIM, MDP3, VideoITG, STTM,
VisionZip) are listed once, under the sweep tables, so the tables are not padded
with rows that cannot vary.

Two ways to run:
    scripts/fetch_results.sh                          # pull results locally
    python3 scripts/build_retention_tables.py --local > docs/RETENTION_TABLES.md

    # or directly on Carya, against $HUVLLM_RESULTS
    python3 build_retention_tables.py > retention_tables.md
"""
import json
import os
import sys

# --local renders from results-cache/ (populated by scripts/fetch_results.sh),
# so tables can be built on a laptop with no cluster access.
if "--local" in sys.argv:
    RES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results-cache")
else:
    RES = os.environ.get("HUVLLM_RESULTS", "/project/rhu/dpalfaro/results")
CATS = ["Action Order", "Camera Motion", "Location-related Motion",
        "Motion Recognition", "Motion-related Objects", "Repetition Count"]

# retention -> {method: run_dir}.  r=0.15 reuses the already-gated wave runs.
STAGE1 = {
    0.10: {"FastV": "s1_fastv_r10_run", "FlashVID": "s1_flashvid_r10_run",
           "HoliTom": "s1_holitom_r10_run", "PruneVID": "s1_prunevid_r10_run"},
    0.15: {"FastV": "w2_fastv_run", "FlashVID": "w2_flashvid_run",
           "HoliTom": "w2_holitom_run", "PruneVID": "w2_prunevid_ov_run"},
    0.25: {"FastV": "s1_fastv_r25_run", "FlashVID": "s1_flashvid_r25_run",
           "HoliTom": "s1_holitom_r25_run", "PruneVID": "s1_prunevid_r25_run"},
}
STAGE3 = {
    0.10: {"FastV": "s3_fastv_r10_run", "FlashVID": "s3_flashvid_r10_run",
           "HoliTom": "s3_holitom_r10_run", "PruneVID": "s3_prunevid_r10_run"},
    0.15: {"FastV": "w3_fastv_run", "FlashVID": "w3_flashvid_run",
           "HoliTom": "w3_holitom_run", "PruneVID": "w3_prunevid_run"},
    0.25: {"FastV": "s3_fastv_r25_run", "FlashVID": "s3_flashvid_r25_run",
           "HoliTom": "s3_holitom_r25_run", "PruneVID": "s3_prunevid_r25_run"},
}
# NOTE: fastv_run1 has fastv_params.enabled=False — it IS the inert LLaVA-OV
# backbone, which is why it serves as the baseline. The real gated FastV
# method run is w2_fastv_run (36.78%). Do not confuse the two.
BASE1, BASE3 = "fastv_run1", "qwen3vl_baseline_run1"
BASE1_ACC, BASE3_ACC = 52.66, 62.52


def load(d):
    p = os.path.join(RES, d)
    try:
        s = json.load(open(os.path.join(p, "summary.json")))
        rows = [json.loads(l) for l in open(os.path.join(p, "results.jsonl"))]
        return s, rows
    except Exception:
        return None, None


def stats(run_dir, base_dir):
    """accuracy, divergence, McNemar chi2 vs the plain backbone, subcategories."""
    s, rows = load(run_dir)
    if s is None:
        return None
    _, brows = load(base_dir)
    div = chi = w = l = None
    if brows:
        n = min(len(rows), len(brows))
        div = sum(1 for i in range(n)
                  if str(rows[i].get("prediction")) != str(brows[i].get("prediction")))
        w = sum(1 for i in range(n)
                if brows[i].get("correct") == 0 and rows[i].get("correct") == 1)
        l = sum(1 for i in range(n)
                if brows[i].get("correct") == 1 and rows[i].get("correct") == 0)
        chi = ((abs(w - l) - 1) ** 2 / (w + l)) if (w + l) else 0.0
    agg = {c: [0, 0] for c in CATS}
    for r in rows:
        c, sc = r.get("question_type"), r.get("correct")
        if c in agg and sc is not None:
            agg[c][1] += 1
            agg[c][0] += sc
    subs = [(100.0 * agg[c][0] / agg[c][1]) if agg[c][1] else None for c in CATS]
    return dict(acc=100.0 * s["accuracy"], correct=s["correct"],
                total=s["total_scoreable"], div=div, chi=chi, w=w, l=l, subs=subs)


def table(title, mapping, base_dir, base_acc, note=""):
    print(f"\n### {title}\n")
    if note:
        print(note + "\n")
    print("| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | "
          "AO | CM | LM | MR | MO | RC |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    print(f"| *baseline* | *{base_acc:.2f}%* | — | *0 (ref)* | — | — | — |"
          + "".join(" *—* |" for _ in CATS))
    got = False
    for m in ("FastV", "FlashVID", "HoliTom", "PruneVID"):
        st = stats(mapping[m], base_dir)
        if st is None:
            print(f"| {m} | 🔄 not yet run | — | — | — | — | — |"
                  + "".join(" — |" for _ in CATS))
            continue
        got = True
        d = st["acc"] - base_acc
        sig = "—" if st["chi"] is None else ("**yes**" if st["chi"] >= 3.84 else "no")
        chis = "—" if st["chi"] is None else f"{st['chi']:.1f}"
        wl = "—" if st["w"] is None else f"{st['w']}/{st['l']}"
        subs = "".join(f" {x:.1f} |" if x is not None else " — |" for x in st["subs"])
        print(f"| **{m}** | **{st['acc']:.2f}%** ({st['correct']}) | {d:+.2f} | "
              f"{st['div']} | {wl} | {chis} | {sig} |{subs}")
    if not got:
        print("\n*(no runs complete at this retention yet)*")


def main():
    print("# Retention sweep — MotionBench\n")
    print("Six tables: two backbones × three retention levels. **Only the "
          "retention knob varies** — 32 frames, greedy decoding and every other "
          "parameter are held fixed, so differences down a column are "
          "attributable to retention alone.\n")
    print("Δ is vs that backbone's plain baseline. χ² is McNemar on paired "
          "predictions (≥3.84 ⇒ p<0.05). *Differ* = predictions ≠ baseline out "
          "of 8,052; **0 would mean the method never engaged**.\n")
    print("Subcategory columns: AO=Action Order, CM=Camera Motion, "
          "LM=Location-related Motion, MR=Motion Recognition, "
          "MO=Motion-related Objects, RC=Repetition Count.\n")
    print("---\n\n## LLaVA-OV-7B (baseline 52.66%)")
    for r in (0.10, 0.15, 0.25):
        table(f"Retention {r:.2f}", STAGE1[r], BASE1, BASE1_ACC)
    print("\n---\n\n## Qwen3-VL-8B (baseline 62.52%)")
    for r in (0.10, 0.15, 0.25):
        table(f"Retention {r:.2f}", STAGE3[r], BASE3, BASE3_ACC)

    # --- the trend view: one row per method, retention across the columns ----
    print("\n---\n\n## Cross-retention summary — does loss track pruning?\n")
    print("The sweep's actual question. Each cell is accuracy (Δ vs that "
          "backbone's baseline); **bold** = statistically significant loss "
          "(McNemar χ² ≥ 3.84).\n")
    for label, mp, bdir, bacc in (("LLaVA-OV-7B", STAGE1, BASE1, BASE1_ACC),
                                  ("Qwen3-VL-8B", STAGE3, BASE3, BASE3_ACC)):
        print(f"\n**{label}** (baseline {bacc:.2f}%)\n")
        print("| Method | r=0.10 | r=0.15 | r=0.25 | trend |")
        print("|---|:---:|:---:|:---:|---|")
        for m in ("FastV", "FlashVID", "HoliTom", "PruneVID"):
            cells, accs = [], []
            for r in (0.10, 0.15, 0.25):
                st = stats(mp[r][m], bdir)
                if st is None:
                    cells.append("🔄"); accs.append(None); continue
                d = st["acc"] - bacc
                sig = st["chi"] is not None and st["chi"] >= 3.84
                txt = f"{st['acc']:.2f}% ({d:+.2f})"
                cells.append(f"**{txt}**" if sig else txt)
                accs.append(st["acc"])
            known = [a for a in accs if a is not None]
            if len(known) < 2:
                trend = "—"
            elif accs[0] is not None and accs[-1] is not None:
                gain = accs[-1] - accs[0]
                trend = (f"+{gain:.2f} from 0.10→0.25 "
                         + ("(more tokens help)" if gain > 0.5 else
                            "(flat — retention barely matters)" if abs(gain) <= 0.5
                            else "(**inverted** — more tokens hurt)"))
            else:
                trend = "partial"
            print(f"| **{m}** | {cells[0]} | {cells[1]} | {cells[2]} | {trend} |")

    print("\n---\n\n## Methods with no retention knob\n")
    print("These do not expose a retention parameter on this axis, so they "
          "appear once rather than in every sweep table. Their mechanism fixes "
          "the token budget internally (or selects frames instead of tokens).\n")
    print("| Method | LLaVA-OV-7B | Qwen3-VL-8B | Why no knob |")
    print("|---|:---:|:---:|---|")
    for m, a, b, why in [
        ("DyCoke", "53.36%", "61.85%", "`p`·`k` are the paper's own two-stage ratios (net ~49%)"),
        ("AIM", "52.86%", "55.97%", "merge schedule compiled into `llava_arch.py`, no CLI knob"),
        ("MDP3", "53.06%", "59.66%", "selects **frames** (32→8), not a token fraction"),
        ("VideoITG", "52.86%", "56.35%", "grounded **frame** selection"),
        ("STTM", "51.72%", "57.07%", "quadtree merges by similarity `--tree_thresh`, not a ratio"),
        ("VisionZip", "40.09%", "58.81%", "fixed `dominant`/`contextual` token counts"),
    ]:
        print(f"| {m} | {a} | {b} | {why} |")

    print("\n---\n\n## Other backbones — each method on its own native model\n")
    print("No retention axis: these are evaluated on the backbone their paper "
          "used, at that paper's setting.\n")
    print("| Method | Backbone | Overall | Status |")
    print("|---|---|:---:|---|")
    print("| PruneVID | PLLaVA-7B | **44.13%** | ✅ its published backbone |")
    print("| DyTo (reconstructed TW-FINCH) | LLaVA-NeXT Vicuna-7B | **42.06%** | "
          "⚠️ gated, but **no divergence check** — no Vicuna baseline yet |")
    print("| VisionZip | LLaVA-1.5-7B | **39.97%** | 🟡 predates the gate |")
    print("| STTM-LLaVAVid | LLaVA-Video-7B | **53.33%** | 🟡 predates the gate |")
    print("| iMove, TrajViT | — | — | ❌ no public code / weights |")


if __name__ == "__main__":
    main()

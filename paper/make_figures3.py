#!/usr/bin/env python3
"""Colored figures for the results-only paper (paper/results.tex).

Palette: validated categorical slots from the dataviz skill's reference
palette (references/palette.md) -- blue #2a78d6 (LLaVA-OV-7B), orange
#eb6834 (Qwen3-VL-8B), aqua #1baf7a / yellow #eda100 for the two extra
method series in the retention plot. Status red #d03b3b marks McNemar
significance. All pairs used here were run through
scripts/validate_palette.js (adjacent pairlist) and passed.

Does not touch fig1-4 (grayscale, used by the full paper.tex) -- writes
new files under figures/ prefixed r3_.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.environ.get("HUVLLM_DATA", os.path.join(_HERE, "data.json"))))
FIG = os.path.join(_HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CATS = ["AO", "CM", "LM", "MR", "MO", "RC"]
CATLABEL = {"AO": "Action\nOrder", "CM": "Camera\nMotion", "LM": "Location\nMotion",
            "MR": "Motion\nRecognition", "MO": "Motion\nObjects", "RC": "Repetition\nCount"}

BLUE = "#2a78d6"      # LLaVA-OV-7B
ORANGE = "#eb6834"    # Qwen3-VL-8B
AQUA = "#1baf7a"
YELLOW = "#eda100"
CRITICAL = "#d03b3b"  # significant (status red)
NEUTRAL = "#c9c7be"   # not significant
INK = "#0b0b0b"
MUTED = "#898781"

ov_base = D["baselines"]["LLaVA-OV-7B"]
qw_base = D["baselines"]["Qwen3-VL-8B"]

# ------------------------------------------------------------------
# Fig r3-1: baseline accuracy by category, colored
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.0))
x = np.arange(len(CATS))
w = 0.38
ov = [ov_base["cats"][c] for c in CATS]
qw = [qw_base["cats"][c] for c in CATS]
ax.bar(x - w/2, ov, w, label=f"LLaVA-OV-7B ({ov_base['overall']:.2f}%)", color=BLUE, edgecolor=INK, linewidth=0.5)
ax.bar(x + w/2, qw, w, label=f"Qwen3-VL-8B ({qw_base['overall']:.2f}%)", color=ORANGE, edgecolor=INK, linewidth=0.5)
ax.axhline(25, color=MUTED, linestyle=":", linewidth=1.0)
ax.text(-0.42, 26.6, "chance (25%)", fontsize=7, ha="left", style="italic", color=MUTED)
for xi, (a, b) in enumerate(zip(ov, qw)):
    ax.text(xi, max(a, b) + 2.0, f"+{b - a:.1f}", ha="center", fontsize=7.5, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels([CATLABEL[c] for c in CATS], fontsize=7.5)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 92)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
ax.set_title("Baseline accuracy by question category", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/r3_fig1_baseline_color.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Fig r3-2: delta vs backbone, both backbones, colored by significance
# ------------------------------------------------------------------
s1 = D["stage1"]
s3 = D["stage3"]
s1_sorted = sorted(s1.items(), key=lambda kv: kv[1]["delta"])
s3_sorted = sorted(s3.items(), key=lambda kv: kv[1]["delta"])

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, items, title in [
    (axes[0], s1_sorted, f"LLaVA-OV-7B (baseline {ov_base['overall']:.2f}%)"),
    (axes[1], s3_sorted, f"Qwen3-VL-8B (baseline {qw_base['overall']:.2f}%)"),
]:
    names = [k for k, _ in items]
    deltas = [v["delta"] for _, v in items]
    sig = [v["significant"] for _, v in items]
    y = np.arange(len(names))
    colors = [CRITICAL if s else NEUTRAL for s in sig]
    ax.barh(y, deltas, color=colors, edgecolor=INK, linewidth=0.6, height=0.62)
    ax.axvline(0, color=INK, linewidth=1)
    ax.axvspan(-1.54, 1.54, color=BLUE, alpha=0.08, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("$\\Delta$ accuracy vs. backbone (pt)")
    ax.set_title(title, fontsize=9)
    for yi, d in zip(y, deltas):
        off = 0.35 if d >= 0 else -0.35
        ax.text(d + off, yi, f"{d:+.2f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=7)

axes[0].set_xlim(-19.5, 4.5)
axes[1].set_xlim(-9.0, 3.0)
handles = [Patch(facecolor=CRITICAL, edgecolor=INK, label="significant (McNemar $\\chi^2\\geq3.84$)"),
           Patch(facecolor=NEUTRAL, edgecolor=INK, label="not significant"),
           Patch(facecolor=BLUE, alpha=0.08, label="$\\pm$1.54 pt resolution floor")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig(f"{FIG}/r3_fig2_delta_color.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Fig r3-3: retention sweep, colored dot+line per method
# ------------------------------------------------------------------
methods = ["FastV", "FlashVID", "HoliTom", "PruneVID"]
mcolor = {"FastV": BLUE, "FlashVID": ORANGE, "HoliTom": AQUA, "PruneVID": YELLOW}
markers = {"FastV": "o", "FlashVID": "s", "HoliTom": "^", "PruneVID": "D"}
LEVELS = {
    "FastV":    {"LLaVA-OV": [0.10, 0.15, 0.25, 0.50, 0.75], "Qwen3-VL": [0.10, 0.15, 0.20, 0.25, 0.50, 0.75]},
    "FlashVID": {"LLaVA-OV": [0.10, 0.15, 0.20, 0.25],       "Qwen3-VL": [0.10, 0.15, 0.20, 0.25]},
    "HoliTom":  {"LLaVA-OV": [0.10, 0.15, 0.20, 0.25],       "Qwen3-VL": [0.10, 0.15, 0.20, 0.25]},
    "PruneVID": {"LLaVA-OV": [0.10, 0.25, 0.50],             "Qwen3-VL": [0.10, 0.25, 0.50]},
}

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
for ax, backbone, base in [(axes[0], "LLaVA-OV", ov_base), (axes[1], "Qwen3-VL", qw_base)]:
    # Categorical (evenly-spaced) x positions, not raw retention values: FastV's
    # range extends to 50/75% while FlashVID/HoliTom stop at 25%, so a true
    # linear axis squeezes 10/15/20/25% into a sliver of the plot width and
    # their tick labels collide. Index position preserves order, not magnitude.
    all_levels = sorted(set(l for m in methods for l in LEVELS[m][backbone]))
    level_to_x = {lvl: i for i, lvl in enumerate(all_levels)}
    for m in methods:
        levels = LEVELS[m][backbone]
        xs = [level_to_x[r] for r in levels]
        ys = [D["retention"][f"{backbone}|{m}|{r:.2f}"]["overall"] for r in levels]
        ax.plot(xs, ys, marker=markers[m], linestyle="-", label=m,
                color=mcolor[m], linewidth=1.6, markersize=6.5,
                markerfacecolor=mcolor[m], markeredgecolor=INK, markeredgewidth=0.6)
    ax.axhline(base["overall"], color=INK, linestyle=(0, (1, 1)), linewidth=1)
    off = 1.3 if backbone == "LLaVA-OV" else 0.6
    ax.text(0.05, base["overall"] + off, f"baseline {base['overall']:.2f}%",
            fontsize=7, ha="left", style="italic", color=MUTED)
    ax.set_xticks(range(len(all_levels)))
    ax.set_xticklabels([f"{int(r*100)}%" for r in all_levels])
    ax.set_xlim(-0.3, len(all_levels) - 0.7)
    ax.set_xlabel("Nominal token retention")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{backbone}-7B" if backbone == "LLaVA-OV" else f"{backbone}-8B", fontsize=9)
axes[0].set_ylim(33, 57)
axes[1].set_ylim(53, 65)
axes[0].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.42))
fig.tight_layout()
fig.savefig(f"{FIG}/r3_fig3_retention_color.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Fig r3-4: Cleveland dot plot -- overall accuracy, all methods, both
# backbones, one figure. This is the "dot graph" of method rankings.
# ------------------------------------------------------------------
all_methods = ["Baseline", "PruneVID", "DyCoke", "HoliTom", "MDP3", "FastV",
               "VisionZip (contextual)", "STTM", "FlashVID", "VideoITG", "AIM"]
# stage1 (LLaVA-OV) labels its PruneVID row "PruneVID-OV"; stage3 (Qwen3-VL)
# calls the same method "PruneVID" -- normalize to one label for the plot.
ov_val = {"Baseline": ov_base["overall"]}
qw_val = {"Baseline": qw_base["overall"]}
for name, e in s1.items():
    ov_val["PruneVID" if name == "PruneVID-OV" else name] = e["overall"]
for name, e in s3.items():
    qw_val[name] = e["overall"]

order = sorted(all_methods, key=lambda m: qw_val.get(m, -1))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
y = np.arange(len(order))
for yi, m in zip(y, order):
    a = ov_val.get(m)
    b = qw_val.get(m)
    if a is not None and b is not None:
        ax.plot([a, b], [yi, yi], color=MUTED, linewidth=1.2, zorder=1)
    if a is not None:
        ax.scatter([a], [yi], color=BLUE, edgecolor=INK, linewidth=0.6, s=55, zorder=2,
                   label="LLaVA-OV-7B" if yi == y[0] else None)
    if b is not None:
        ax.scatter([b], [yi], color=ORANGE, edgecolor=INK, linewidth=0.6, s=55, zorder=2,
                   label="Qwen3-VL-8B" if yi == y[0] else None)
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=8.5)
ax.set_xlabel("Overall accuracy (%)")
ax.set_title("Overall accuracy by method and backbone (15% nominal retention)", fontsize=9.5)
ax.axvline(25, color=MUTED, linestyle=":", linewidth=0.8)
ax.text(25.3, -0.7, "chance (25%)", fontsize=6.5, style="italic", color=MUTED)
ax.set_ylim(-0.8, len(order) - 0.2)
handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=INK, markersize=7, label="LLaVA-OV-7B"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor=INK, markersize=7, label="Qwen3-VL-8B")]
ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIG}/r3_fig4_dotplot_methods.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote colored figures")
for f in sorted(os.listdir(FIG)):
    if f.startswith("r3_"):
        print(" ", f)

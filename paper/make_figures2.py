#!/usr/bin/env python3
"""Figures for the revised paper. Restricted to overall accuracy, per-category
accuracy, and delta vs baseline -- no divergence/chi-square in any graphic."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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

DARK = "#1a1a1a"
MID = "#7a7a7a"
LIGHT = "#c4c4c4"

ov_base = D["baselines"]["LLaVA-OV-7B"]
qw_base = D["baselines"]["Qwen3-VL-8B"]

# ------------------------------------------------------------------
# Figure 1 -- baseline capability profile across categories
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.0))
x = np.arange(len(CATS))
w = 0.38
ov = [ov_base["cats"][c] for c in CATS]
qw = [qw_base["cats"][c] for c in CATS]
ax.bar(x - w/2, ov, w, label=f"LLaVA-OV-7B ({ov_base['overall']:.2f}%)", color=LIGHT, edgecolor=DARK, linewidth=0.6)
ax.bar(x + w/2, qw, w, label=f"Qwen3-VL-8B ({qw_base['overall']:.2f}%)", color=DARK)
ax.axhline(25, color="black", linestyle=":", linewidth=0.9)
ax.text(-0.42, 26.6, "chance (25%)", fontsize=7, ha="left", style="italic")
for xi, (a, b) in enumerate(zip(ov, qw)):
    ax.text(xi, max(a, b) + 2.0, f"+{b - a:.1f}", ha="center", fontsize=7.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([CATLABEL[c] for c in CATS], fontsize=7.5)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 92)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
ax.set_title("Baseline accuracy by question category (n as annotated in Table 1)", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_baseline_profile.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Figure 2 -- delta vs baseline, both backbones (the transfer finding)
# ------------------------------------------------------------------
s1 = D["stage1"]
s3 = D["stage3"]
s1_sorted = sorted(s1.items(), key=lambda kv: kv[1]["delta"])
s3_sorted = sorted(s3.items(), key=lambda kv: kv[1]["delta"])

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

for ax, items, title, base in [
    (axes[0], s1_sorted, f"LLaVA-OV-7B (baseline {ov_base['overall']:.2f}%)", ov_base),
    (axes[1], s3_sorted, f"Qwen3-VL-8B (baseline {qw_base['overall']:.2f}%)", qw_base),
]:
    names = [k for k, _ in items]
    deltas = [v["delta"] for _, v in items]
    sig = [v["significant"] for _, v in items]
    y = np.arange(len(names))
    colors = [DARK if s else LIGHT for s in sig]
    ax.barh(y, deltas, color=colors, edgecolor=DARK, linewidth=0.6, height=0.62)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvspan(-1.54, 1.54, color="#000000", alpha=0.07, zorder=0)
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
handles = [Patch(facecolor=DARK, edgecolor=DARK, label="significant (McNemar $\\chi^2\\geq3.84$)"),
           Patch(facecolor=LIGHT, edgecolor=DARK, label="not significant"),
           Patch(facecolor="#000000", alpha=0.07, label="$\\pm$1.54 pt resolution floor")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_delta_both.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Figure 3 -- per-category delta heatmap
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5),
                          gridspec_kw={"width_ratios": [len(s1), len(s3)], "wspace": 0.08})

for ax, items, title in [(axes[0], s1_sorted[::-1], "LLaVA-OV-7B"),
                          (axes[1], s3_sorted[::-1], "Qwen3-VL-8B")]:
    names = [k for k, _ in items]
    M = np.array([[v["cat_delta"][c] for c in CATS] for _, v in items])
    im = ax.imshow(M.T, cmap="RdBu", vmin=-20, vmax=20, aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(len(CATS)))
    if ax is axes[0]:
        ax.set_yticklabels([CATLABEL[c].replace("\n", " ") for c in CATS], fontsize=7.5)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=9)
    for i in range(len(names)):
        for j in range(len(CATS)):
            v = M[i, j]
            ax.text(i, j, f"{v:+.1f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(v) > 11 else "black")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CATS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label("$\\Delta$ accuracy vs. backbone (pt)", fontsize=8)
cbar.ax.tick_params(labelsize=7)
fig.suptitle("Per-category change vs. backbone, by method", fontsize=9.5, y=1.0)
fig.savefig(f"{FIG}/fig3_category_delta.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Figure 4 -- retention sweep (overall accuracy only)
# ------------------------------------------------------------------
methods = ["FastV", "FlashVID", "HoliTom", "PruneVID"]
LEVELS = {
    "FastV":    {"LLaVA-OV": [0.10, 0.15, 0.25, 0.50, 0.75], "Qwen3-VL": [0.10, 0.15, 0.25]},
    "FlashVID": {"LLaVA-OV": [0.10, 0.15, 0.20, 0.25],       "Qwen3-VL": [0.10, 0.15, 0.25]},
    "HoliTom":  {"LLaVA-OV": [0.10, 0.15, 0.20, 0.25],       "Qwen3-VL": [0.10, 0.15, 0.25]},
    "PruneVID": {"LLaVA-OV": [0.10, 0.25, 0.50],             "Qwen3-VL": [0.10, 0.25, 0.50]},
}
markers = {"FastV": "o", "FlashVID": "s", "HoliTom": "^", "PruneVID": "D"}
styles = {"FastV": "-", "FlashVID": "--", "HoliTom": "-.", "PruneVID": ":"}

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
for ax, backbone, base in [(axes[0], "LLaVA-OV", ov_base), (axes[1], "Qwen3-VL", qw_base)]:
    for m in methods:
        levels = LEVELS[m][backbone]
        ys = [D["retention"][f"{backbone}|{m}|{r:.2f}"]["overall"] for r in levels]
        ax.plot(levels, ys, marker=markers[m], linestyle=styles[m], label=m,
                color=DARK, linewidth=1.3, markersize=5, markerfacecolor="white")
    ax.axhline(base["overall"], color="black", linestyle=(0, (1, 1)), linewidth=1)
    off = 1.3 if backbone == "LLaVA-OV" else 0.6
    ax.text(0.098, base["overall"] + off, f"baseline {base['overall']:.2f}%",
            fontsize=7, ha="left", style="italic")
    all_levels = sorted(set(l for m in methods for l in LEVELS[m][backbone]))
    ax.set_xticks(all_levels)
    ax.set_xticklabels([f"{int(r*100)}%" for r in all_levels])
    ax.set_xlabel("Nominal token retention")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{backbone}-7B" if backbone == "LLaVA-OV" else f"{backbone}-8B", fontsize=9)
axes[0].set_ylim(33, 57)
axes[1].set_ylim(53, 65)
axes[0].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.42))
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_retention.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote figures")
for f in sorted(os.listdir(FIG)):
    print(" ", f)

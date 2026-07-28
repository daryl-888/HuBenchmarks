# Retention sweep — MotionBench

Six tables: two backbones × three retention levels. **Only the retention knob varies** — 32 frames, greedy decoding and every other parameter are held fixed, so differences down a column are attributable to retention alone.

Δ is vs that backbone's plain baseline. χ² is McNemar on paired predictions (≥3.84 ⇒ p<0.05). *Differ* = predictions ≠ baseline out of 8,052; **0 would mean the method never engaged**.

Subcategory columns: AO=Action Order, CM=Camera Motion, LM=Location-related Motion, MR=Motion Recognition, MO=Motion-related Objects, RC=Repetition Count.

---

## LLaVA-OV-7B (baseline 52.66%)

### Retention 0.10

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *52.66%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| FastV | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| FlashVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| HoliTom | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| PruneVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |

*(no runs complete at this retention yet)*

### Retention 0.15

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *52.66%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| **FastV** | **36.78%** (1478) | -15.88 | 4605 | 475/1113 | 255.5 | **yes** | 32.8 | 31.9 | 34.1 | 35.7 | 53.8 | 25.2 |
| **FlashVID** | **53.31%** (2142) | +0.65 | 1610 | 268/242 | 1.2 | no | 39.9 | 45.7 | 53.8 | 58.0 | 71.7 | 28.2 |
| **HoliTom** | **53.14%** (2135) | +0.48 | 1838 | 299/280 | 0.6 | no | 40.8 | 49.6 | 52.6 | 57.0 | 71.4 | 27.2 |
| **PruneVID** | **38.20%** (1535) | -14.46 | 4536 | 473/1054 | 220.3 | **yes** | 32.0 | 34.8 | 35.5 | 38.4 | 52.9 | 27.2 |

### Retention 0.25

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *52.66%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| FastV | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| FlashVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| HoliTom | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| PruneVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |

*(no runs complete at this retention yet)*

---

## Qwen3-VL-8B (baseline 62.52%)

### Retention 0.10

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *62.52%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| FastV | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| FlashVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| HoliTom | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| PruneVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |

*(no runs complete at this retention yet)*

### Retention 0.15

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *62.52%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| **FastV** | **59.01%** (2371) | -3.51 | 2029 | 149/290 | 44.6 | **yes** | 46.2 | 61.0 | 61.9 | 63.5 | 72.2 | 30.5 |
| **FlashVID** | **56.65%** (2276) | -5.87 | 2625 | 178/414 | 93.3 | **yes** | 43.4 | 57.9 | 57.1 | 61.2 | 73.0 | 27.0 |
| **HoliTom** | **60.33%** (2424) | -2.19 | 1657 | 131/219 | 21.6 | **yes** | 44.7 | 61.3 | 61.0 | 66.4 | 76.2 | 29.0 |
| **PruneVID** | **62.17%** (2498) | -0.35 | 1074 | 56/70 | 1.3 | no | 46.1 | 62.9 | 64.1 | 67.1 | 79.1 | 32.5 |

### Retention 0.25

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *62.52%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| FastV | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| FlashVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| HoliTom | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |
| PruneVID | 🔄 not yet run | — | — | — | — | — | — | — | — | — | — | — |

*(no runs complete at this retention yet)*

---

## Cross-retention summary — does loss track pruning?

The sweep's actual question. Each cell is accuracy (Δ vs that backbone's baseline); **bold** = statistically significant loss (McNemar χ² ≥ 3.84).


**LLaVA-OV-7B** (baseline 52.66%)

| Method | r=0.10 | r=0.15 | r=0.25 | trend |
|---|:---:|:---:|:---:|---|
| **FastV** | 🔄 | **36.78% (-15.88)** | 🔄 | — |
| **FlashVID** | 🔄 | 53.31% (+0.65) | 🔄 | — |
| **HoliTom** | 🔄 | 53.14% (+0.48) | 🔄 | — |
| **PruneVID** | 🔄 | **38.20% (-14.46)** | 🔄 | — |

**Qwen3-VL-8B** (baseline 62.52%)

| Method | r=0.10 | r=0.15 | r=0.25 | trend |
|---|:---:|:---:|:---:|---|
| **FastV** | 🔄 | **59.01% (-3.51)** | 🔄 | — |
| **FlashVID** | 🔄 | **56.65% (-5.87)** | 🔄 | — |
| **HoliTom** | 🔄 | **60.33% (-2.19)** | 🔄 | — |
| **PruneVID** | 🔄 | 62.17% (-0.35) | 🔄 | — |

---

## Methods with no retention knob

These do not expose a retention parameter on this axis, so they appear once rather than in every sweep table. Their mechanism fixes the token budget internally (or selects frames instead of tokens).

| Method | LLaVA-OV-7B | Qwen3-VL-8B | Why no knob |
|---|:---:|:---:|---|
| DyCoke | 53.36% | 61.85% | `p`·`k` are the paper's own two-stage ratios (net ~49%) |
| AIM | 52.86% | 55.97% | merge schedule compiled into `llava_arch.py`, no CLI knob |
| MDP3 | 53.06% | 59.66% | selects **frames** (32→8), not a token fraction |
| VideoITG | 52.86% | 56.35% | grounded **frame** selection |
| STTM | 51.72% | 57.07% | quadtree merges by similarity `--tree_thresh`, not a ratio |
| VisionZip | 40.09% | 58.81% | fixed `dominant`/`contextual` token counts |

---

## Other backbones — each method on its own native model

No retention axis: these are evaluated on the backbone their paper used, at that paper's setting.

| Method | Backbone | Overall | Status |
|---|---|:---:|---|
| PruneVID | PLLaVA-7B | **44.13%** | ✅ its published backbone |
| DyTo (reconstructed TW-FINCH) | LLaVA-NeXT Vicuna-7B | **42.06%** | ⚠️ gated, but **no divergence check** — no Vicuna baseline yet |
| VisionZip | LLaVA-1.5-7B | **39.97%** | 🟡 predates the gate |
| STTM-LLaVAVid | LLaVA-Video-7B | **53.33%** | 🟡 predates the gate |
| iMove, TrajViT | — | — | ❌ no public code / weights |

# Retention sweep — MotionBench


> **This is the standalone retention sweep.** For the combined report — headline findings, gated results, this sweep, the sampled study and the DyTo control in one place — see [FINDINGS.md](FINDINGS.md).

Six tables: two backbones × three retention levels. **Only the retention knob varies** — 32 frames, greedy decoding and every other parameter are held fixed, so differences down a column are attributable to retention alone.

Δ is vs that backbone's plain baseline. χ² is McNemar on paired predictions (≥3.84 ⇒ p<0.05). *Differ* = predictions ≠ baseline out of 8,052; **0 would mean the method never engaged**.

Subcategory columns: AO=Action Order, CM=Camera Motion, LM=Location-related Motion, MR=Motion Recognition, MO=Motion-related Objects, RC=Repetition Count.

---

## LLaVA-OV-7B (baseline 52.66%)

### Retention 0.10

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *52.66%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| **FastV** | **36.06%** (1449) | -16.60 | 4656 | 462/1129 | 278.8 | **yes** | 32.9 | 31.9 | 33.9 | 34.1 | 53.3 | 24.5 |
| **FlashVID** | **52.51%** (2110) | -0.15 | 1937 | 303/309 | 0.0 | no | 41.0 | 46.8 | 53.3 | 56.3 | 70.7 | 26.5 |
| **HoliTom** | **52.76%** (2120) | +0.10 | 2071 | 336/332 | 0.0 | no | 40.5 | 48.8 | 53.3 | 55.8 | 71.2 | 29.0 |
| **PruneVID** | **38.10%** (1531) | -14.56 | 4682 | 501/1086 | 214.9 | **yes** | 33.5 | 33.5 | 36.8 | 38.2 | 52.3 | 25.2 |

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
| **FastV** | **36.73%** (1476) | -15.93 | 4570 | 473/1113 | 257.5 | **yes** | 32.0 | 31.7 | 33.3 | 36.3 | 53.0 | 26.0 |
| **FlashVID** | **53.36%** (2144) | +0.70 | 1473 | 226/198 | 1.7 | no | 42.2 | 47.5 | 55.1 | 57.5 | 71.9 | 23.8 |
| **HoliTom** | **53.41%** (2146) | +0.75 | 1485 | 250/220 | 1.8 | no | 40.7 | 48.3 | 54.8 | 57.7 | 72.3 | 24.5 |
| **PruneVID** | **38.38%** (1542) | -14.28 | 4612 | 486/1060 | 212.4 | **yes** | 32.2 | 33.2 | 36.6 | 38.8 | 53.5 | 26.2 |

---

## Qwen3-VL-8B (baseline 62.52%)

### Retention 0.10

| Method | Overall | Δ vs base | Differ | W/L | χ² | Sig? | AO | CM | LM | MR | MO | RC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *baseline* | *62.52%* | — | *0 (ref)* | — | — | — | *—* | *—* | *—* | *—* | *—* | *—* |
| **FastV** | **56.92%** (2287) | -5.60 | 2466 | 167/392 | 89.8 | **yes** | 44.5 | 60.5 | 59.0 | 61.4 | 69.9 | 28.0 |
| **FlashVID** | **54.73%** (2199) | -7.79 | 3097 | 195/508 | 138.5 | **yes** | 44.5 | 54.5 | 56.0 | 58.4 | 70.9 | 25.0 |
| **HoliTom** | **60.05%** (2413) | -2.47 | 1761 | 124/223 | 27.7 | **yes** | 44.9 | 60.8 | 59.7 | 65.7 | 77.4 | 28.8 |
| **PruneVID** | **60.23%** (2420) | -2.29 | 2422 | 156/248 | 20.5 | **yes** | 46.6 | 60.8 | 61.0 | 65.2 | 77.1 | 28.8 |

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
| **FastV** | **60.60%** (2435) | -1.92 | 1538 | 120/197 | 18.2 | **yes** | 46.4 | 62.3 | 63.7 | 65.2 | 74.1 | 32.8 |
| **FlashVID** | **59.36%** (2385) | -3.16 | 2007 | 155/282 | 36.3 | **yes** | 43.9 | 62.3 | 61.0 | 63.8 | 75.7 | 29.8 |
| **HoliTom** | **60.43%** (2428) | -2.09 | 1478 | 120/204 | 21.3 | **yes** | 45.1 | 61.0 | 61.0 | 66.0 | 77.5 | 28.8 |
| **PruneVID** | **61.40%** (2467) | -1.12 | 1705 | 94/139 | 8.3 | **yes** | 46.1 | 60.0 | 63.0 | 66.6 | 78.7 | 31.5 |

---

## Cross-retention summary — does loss track pruning?

The sweep's actual question. Each cell is accuracy (Δ vs that backbone's baseline); **bold** = statistically significant loss (McNemar χ² ≥ 3.84).


**LLaVA-OV-7B** (baseline 52.66%)

| Method | r=0.10 | r=0.15 | r=0.25 | trend |
|---|:---:|:---:|:---:|---|
| **FastV** | **36.06% (-16.60)** | **36.78% (-15.88)** | **36.73% (-15.93)** | +0.67 from 0.10→0.25 (more tokens help) |
| **FlashVID** | 52.51% (-0.15) | 53.31% (+0.65) | 53.36% (+0.70) | +0.85 from 0.10→0.25 (more tokens help) |
| **HoliTom** | 52.76% (+0.10) | 53.14% (+0.48) | 53.41% (+0.75) | +0.65 from 0.10→0.25 (more tokens help) |
| **PruneVID** | **38.10% (-14.56)** | **38.20% (-14.46)** | **38.38% (-14.28)** | +0.27 from 0.10→0.25 (flat — retention barely matters) |

**Qwen3-VL-8B** (baseline 62.52%)

| Method | r=0.10 | r=0.15 | r=0.25 | trend |
|---|:---:|:---:|:---:|---|
| **FastV** | **56.92% (-5.60)** | **59.01% (-3.51)** | **60.60% (-1.92)** | +3.68 from 0.10→0.25 (more tokens help) |
| **FlashVID** | **54.73% (-7.79)** | **56.65% (-5.87)** | **59.36% (-3.16)** | +4.63 from 0.10→0.25 (more tokens help) |
| **HoliTom** | **60.05% (-2.47)** | **60.33% (-2.19)** | **60.43% (-2.09)** | +0.37 from 0.10→0.25 (flat — retention barely matters) |
| **PruneVID** | **60.23% (-2.29)** | 62.17% (-0.35) | **61.40% (-1.12)** | +1.17 from 0.10→0.25 (more tokens help) |

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

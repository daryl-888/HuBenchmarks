# Results

MotionBench: 8,052 questions, **4,018 scoreable** (4,034 `NA` excluded) — the
denominator for every accuracy below. Greedy decoding, 32 frames. Categories:
**AO** Action Order · **CM** Camera Motion · **LM** Location-related Motion ·
**MR** Motion Recognition · **MO** Motion-related Objects · **RC** Repetition Count.
Chance is 25%.

Differences below **±1.54 pt** are not resolvable at this n and are not ranked.
`*` marks significance by McNemar on paired predictions (χ² ≥ 3.84, p<0.05).
Every figure is computed from per-sample predictions by
[`scripts/build_results_tables.py`](../scripts/build_results_tables.py) — do not hand-edit.

| Backbone | Baseline | AO | CM | LM | MR | MO | RC |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-OV-7B | 52.66% | 40.5 | 45.2 | 55.5 | 57.0 | 71.2 | 23.8 |
| Qwen3-VL-8B | 62.52% | 46.1 | 63.1 | 65.0 | 67.3 | 79.0 | 33.8 |

## 1. Every method in its original configuration

Each method at the setting its authors published, on the backbone it was
designed for where one exists. Methods whose published setting is 15%
retention appear at 15%; others at their own default, stated per row.

| Method | Venue | Year | Backbone | Setting | Overall | Δ base | AO | CM | LM | MR | MO | RC |
|---|:--:|:--:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FlashVID** | ICLR (Oral) | 2026 | LLaVA-OV | retention=0.25 | **53.36%** | +0.70 | 42.2 | 47.5 | 55.1 | 57.5 | 71.9 | 23.8 |
| **DyCoke** | arXiv | 2024 | LLaVA-OV | l=3 p=0.7 k=0.7 | **53.36%** | +0.70 | 38.9 | 48.8 | 55.7 | 58.2 | 70.9 | 25.3 |
| **HoliTom** | — | 2025 | LLaVA-OV | RETAIN=0.15 T=0.80 k=18 r=0.5 | **53.14%** | +0.47 | 40.8 | 49.6 | 52.6 | 57.0 | 71.4 | 27.3 |
| **MDP3** | ICCV | 2025 | LLaVA-OV | pool=32 select=8 | **53.06%** | +0.40 | 40.5 | 49.6 | 53.5 | 56.8 | 71.6 | 26.5 |
| **VideoITG** | — | 2025 | LLaVA-OV | 512 sampled / 32 selected | **52.86%** | +0.20 | 40.1 | 47.0 | 53.7 | 57.6 | 70.1 | 26.8 |
| **AIM** | ICCV | 2025 | LLaVA-OV | 4-step bipartite merge + PageRank | **52.86%** | +0.20 | 41.4 | 48.1 | 54.2 | 57.0 | 71.9 | 22.3 |
| **STTM** | — | 2025 | LLaVA-OV | thresh=0.85 temporal=0.65 root=1 | **51.72%** | −0.95 | 39.9 | 48.1 | 53.8 | 53.5 | 70.6 | 28.8 |
| **DyTo**¹ | ICCV | 2025 | Vicuna-7B | spatial_tome_finch_dynamic | **42.06%** | — | 32.4 | 33.0 | 42.1 | 43.0 | 61.7 | 26.0 |
| **VisionZip**² | — | 2024 | LLaVA-1.5-7B | dominant=54 contextual=10, 8f | **40.09%** | — | 33.7 | 33.0 | 37.2 | 41.1 | 57.4 | 25.8 |
| **PruneVID**³ | — | 2024 | LLaVA-OV | cluster=0.50 seg=0.25 | **38.20%** | −14.46* | 32.0 | 34.8 | 35.5 | 38.4 | 52.9 | 27.3 |
| **FastV**⁴ | arXiv | 2024 | LLaVA-OV | keep 15% | **36.78%** | −15.88* | 32.8 | 31.9 | 34.1 | 35.7 | 53.8 | 25.3 |

¹ **DyTo** — native backbone; no baseline.  
² **VisionZip** — native backbone.  
³ **PruneVID** — collapses at the published default.  
⁴ **FastV** — no video config released; see note.  

## 2. LLaVA-OV-7B

All methods on this backbone at 15% nominal retention where the method
exposes one; frame-selection methods keep their defaults.

| Method | Venue | Year | Overall | Δ base | AO | CM | LM | MR | MO | RC |
|---|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| DyCoke | arXiv | 2024 | **53.36%** | +0.70 | 38.9 | 48.8 | 55.7 | 58.2 | 70.9 | 25.3 |
| FlashVID | ICLR (Oral) | 2026 | **53.31%** | +0.65 | 39.9 | 45.7 | 53.8 | 58.0 | 71.7 | 28.3 |
| HoliTom | — | 2025 | **53.14%** | +0.47 | 40.8 | 49.6 | 52.6 | 57.0 | 71.4 | 27.3 |
| MDP3 | ICCV | 2025 | **53.06%** | +0.40 | 40.5 | 49.6 | 53.5 | 56.8 | 71.6 | 26.5 |
| VideoITG | — | 2025 | **52.86%** | +0.20 | 40.1 | 47.0 | 53.7 | 57.6 | 70.1 | 26.8 |
| AIM | ICCV | 2025 | **52.86%** | +0.20 | 41.4 | 48.1 | 54.2 | 57.0 | 71.9 | 22.3 |
| *backbone* | — | — | *52.66%* | — | 40.5 | 45.2 | 55.5 | 57.0 | 71.2 | 23.8 |
| STTM | — | 2025 | **51.72%** | −0.95 | 39.9 | 48.1 | 53.8 | 53.5 | 70.6 | 28.8 |
| PruneVID | — | 2024 | **38.20%** | −14.46* | 32.0 | 34.8 | 35.5 | 38.4 | 52.9 | 27.3 |
| FastV | arXiv | 2024 | **36.78%** | −15.88* | 32.8 | 31.9 | 34.1 | 35.7 | 53.8 | 25.3 |

## 3. Qwen3-VL-8B

All methods on this backbone at 15% nominal retention where the method
exposes one; frame-selection methods keep their defaults.

| Method | Venue | Year | Overall | Δ base | AO | CM | LM | MR | MO | RC |
|---|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| *backbone* | — | — | *62.52%* | — | 46.1 | 63.1 | 65.0 | 67.3 | 79.0 | 33.8 |
| PruneVID | — | 2024 | **62.17%** | −0.35 | 46.1 | 62.9 | 64.1 | 67.1 | 79.1 | 32.5 |
| DyCoke | arXiv | 2024 | **61.85%** | −0.67 | 45.1 | 62.6 | 63.9 | 66.4 | 78.4 | 34.5 |
| HoliTom | — | 2025 | **60.33%** | −2.19* | 44.7 | 61.3 | 61.0 | 66.4 | 76.2 | 29.0 |
| MDP3 | ICCV | 2025 | **59.66%** | −2.86* | 44.9 | 64.7 | 62.8 | 64.0 | 77.4 | 23.0 |
| FastV | arXiv | 2024 | **59.01%** | −3.51* | 46.2 | 61.0 | 61.9 | 63.5 | 72.2 | 30.5 |
| VisionZip | — | 2024 | **58.81%** | −3.71* | 43.2 | 57.4 | 61.9 | 64.2 | 74.3 | 29.5 |
| STTM | — | 2025 | **57.07%** | −5.45* | 44.3 | 60.0 | 61.0 | 60.6 | 73.8 | 23.8 |
| FlashVID | ICLR (Oral) | 2026 | **56.65%** | −5.87* | 43.4 | 57.9 | 57.1 | 61.2 | 73.0 | 27.0 |
| VideoITG | — | 2025 | **56.35%** | −6.17* | 45.3 | 53.5 | 59.3 | 60.2 | 75.4 | 22.3 |
| AIM | ICCV | 2025 | **55.97%** | −6.55* | 45.9 | 56.6 | 59.5 | 58.3 | 72.2 | 27.0 |

## 4. Retention sweep — 0.10 / 0.15 / 0.25

**Only methods that remain intact at 0.15 and were actually run at all three**
**ratios appear here.** FastV is excluded: it collapses across this entire band
(36.06 / 36.78 / 36.73 on LLaVA-OV) and its numbers describe a failure mode, not
a retention response. PruneVID is excluded: it has no 0.15 measurement — the cell
previously labelled 0.15 was its published `cluster_ratio=0.50`. Both are
diagnosed in [RETENTION_DIAGNOSIS.md](RETENTION_DIAGNOSIS.md).

### LLaVA-OV-7B (baseline 52.66%)

| Method | Venue | Year | r | Overall | Δ base | AO | CM | LM | MR | MO | RC |
|---|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FlashVID** | ICLR (Oral) | 2026 | 0.10 | 52.51% | −0.15 | 41.0 | 46.8 | 53.3 | 56.3 | 70.7 | 26.5 |
|  |  |  | 0.15 | 53.31% | +0.65 | 39.9 | 45.7 | 53.8 | 58.0 | 71.7 | 28.3 |
|  |  |  | 0.25 | 53.36% | +0.70 | 42.2 | 47.5 | 55.1 | 57.5 | 71.9 | 23.8 |
| **HoliTom** | — | 2025 | 0.10 | 52.76% | +0.10 | 40.5 | 48.8 | 53.3 | 55.8 | 71.2 | 29.0 |
|  |  |  | 0.15 | 53.14% | +0.47 | 40.8 | 49.6 | 52.6 | 57.0 | 71.4 | 27.3 |
|  |  |  | 0.25 | 53.41% | +0.75 | 40.7 | 48.3 | 54.8 | 57.7 | 72.3 | 24.5 |

### Qwen3-VL-8B (baseline 62.52%)

| Method | Venue | Year | r | Overall | Δ base | AO | CM | LM | MR | MO | RC |
|---|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FlashVID** | ICLR (Oral) | 2026 | 0.10 | 54.73% | −7.79* | 44.5 | 54.5 | 56.0 | 58.4 | 70.9 | 25.0 |
|  |  |  | 0.15 | 56.65% | −5.87* | 43.4 | 57.9 | 57.1 | 61.2 | 73.0 | 27.0 |
|  |  |  | 0.25 | 59.36% | −3.16* | 43.9 | 62.3 | 61.0 | 63.8 | 75.7 | 29.8 |
| **HoliTom** | — | 2025 | 0.10 | 60.05% | −2.46* | 44.9 | 60.8 | 59.7 | 65.7 | 77.4 | 28.8 |
|  |  |  | 0.15 | 60.33% | −2.19* | 44.7 | 61.3 | 61.0 | 66.4 | 76.2 | 29.0 |
|  |  |  | 0.25 | 60.43% | −2.09* | 45.1 | 61.0 | 61.0 | 66.0 | 77.5 | 28.8 |

## 5. Methods on their own backbones

Bound to a backbone neither standardized model can host. **No Δ is given:**
no matched baseline exists, so these are absolute scores and are not
comparable to the tables above.

| Method | Venue | Year | Backbone | Setting | Overall | AO | CM | LM | MR | MO | RC |
|---|:--:|:--:|---|---|---:|---:|---:|---:|---:|---:|---:|
| VisionZip | — | 2024 | LLaVA-1.5-7B | 8 frames, dominant=54 contextual=10 | **40.09%** | 33.7 | 33.0 | 37.2 | 41.1 | 57.4 | 25.8 |
| DyTo | ICCV | 2025 | LLaVA-NeXT Vicuna-7B | reconstructed TW-FINCH | **42.06%** | 32.4 | 33.0 | 42.1 | 43.0 | 61.7 | 26.0 |

VisionZip patches `CLIPVisionTower`, which LLaVA-OV does not have, so its
complete form runs only on LLaVA-1.5-7B and at 8 frames. DyTo is not
reproducible from its published artifacts and is reported as
*DyTo (reconstructed TW-FINCH)*; it is also the one number here with
execution evidence but no divergence check, since no baseline exists on its
backbone. See [UPSTREAM_DEFECTS.md](UPSTREAM_DEFECTS.md).

---

Related: [RETENTION_DIAGNOSIS.md](RETENTION_DIAGNOSIS.md) ·
[FASTV_COLLAPSE_ANALYSIS.md](FASTV_COLLAPSE_ANALYSIS.md) ·
[METHODOLOGY.md](METHODOLOGY.md) · [DETERMINISM_AND_VALIDITY.md](DETERMINISM_AND_VALIDITY.md)

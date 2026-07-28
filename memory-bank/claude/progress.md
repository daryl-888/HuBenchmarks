# Progress — HuBenchmarks (2026-07-28)

Numbers here are a summary. **[master-results.md](master-results.md) is
authoritative**; if they disagree, that file wins.

## Status: matrix complete, one reference run outstanding

| Track | Gated | Baseline |
|---|:---:|:---:|
| LLaVA-OV-7B (Stage 1) | 10/11 | 52.66% |
| Qwen3-VL-8B (Stage 3) | **10/10 portable** | 62.52% |
| DyTo (own Vicuna backbone) | 1 (no divergence check) | — |

**Outstanding:** job 7882502, vanilla Vicuna baseline. Closes DyTo's missing
`--vs-baseline` reference. Nothing else depends on it.

## Stage 1 — LLaVA-OV-7B (baseline 52.66%)

| Method | Acc | Differ |
|---|:---:|:---:|
| DyCoke | 53.36% | 1031 |
| FlashVID | 53.31% | 1610 |
| HoliTom | 53.14% | 1838 |
| MDP3 | 53.06% | 1452 |
| AIM | 52.86% | 1406 |
| VideoITG | 52.86% | 1193 |
| STTM | 51.72% | 4859 |
| VisionZip | 40.09% | 4506 |
| PruneVID (OV port) | 38.20% | 4536 |
| FastV | 36.78% | 4605 |

**No method significantly beats the backbone.** Only FastV and VisionZip differ
significantly and both are *worse* — both were run at the standardized 15%
retention, far more aggressive than their paper defaults.

## Stage 3 — Qwen3-VL-8B (baseline 62.52%)

| Method | Acc | Δ | χ² | Sig? |
|---|:---:|:---:|:---:|:---:|
| PruneVID | 62.17% | −0.35 | 1.3 | no |
| DyCoke | 61.85% | −0.67 | 3.5 | no |
| HoliTom | 60.33% | −2.19 | 21.6 | yes |
| MDP3 | 59.66% | −2.86 | 30.0 | yes |
| FastV | 59.01% | −3.51 | 44.6 | yes |
| VisionZip (contextual-only) | 58.81% | −3.71 | 41.3 | yes |
| STTM | 57.07% | −5.45 | 85.0 | yes |
| FlashVID | 56.65% | −5.87 | 93.3 | yes |
| VideoITG | 56.35% | −6.17 | 92.4 | yes |
| AIM | 55.97% | −6.55 | 105.4 | yes |

**All 10 lose; 8 significantly.** The two non-significant losses are the two most
conservative methods (50% and 49% retention vs 15% for the rest).

## Other backbones

| Method | Backbone | Acc | Note |
|---|---|:---:|---|
| PruneVID | PLLaVA-7B | 44.13% | its published backbone |
| DyTo (reconstructed TW-FINCH) | Vicuna-7B | 42.06% | ⚠️ no divergence check — see activeContext |
| VisionZip | LLaVA-1.5-7B | 39.97% | 🟡 predates the gate |
| STTM-LLaVAVid | LLaVA-Video-7B | 53.33% | 🟡 predates the gate |
| iMove, TrajViT | — | — | no public code / weights |

## Not runnable / descoped

* **Stage 2 (LLaVA-Video-7B)** — descoped 2026-07-23.
* **DyTo on Qwen3-VL** — bound to LLaVA's multimodal pipeline; runs on Vicuna.
* **iMove / TrajViT** — retrieval-only or no released weights.

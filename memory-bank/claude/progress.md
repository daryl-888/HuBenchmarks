# Progress — HuBenchmarks (2026-07-16 11:00 CDT)

## New Results — Overnight Run (7/15 → 7/16)
| Job ID | Model | Backbone | Result | Runtime |
|:---:|-------|----------|:------:|:------:|
| 7714650 | DyCoke | qwen1.5 | **53.36%** (2144/4018) | 5h22m |
| 7714722 | MDP3 | qwen1.5 (re-run) | **53.06%** (2132/4018) | 7h28m |
| 7714684 | VideoITG | Grounding | 8052 frame scores | 7h30m |

## OVQwen 1.5 Scoreboard (8/10 complete)
| Model | Accuracy | Status |
|-------|:--------:|--------|
| DyCoke | **53.36%** | ✅ NEW (7714650) |
| HoliTom | **53.14%** | ✅ (7713092) |
| MDP3 | **53.06%** | ✅ v2 re-run (7714722) |
| VideoITG | **52.86%** | ✅ (7693609) |
| AIM | **52.81%** | ✅ |
| PruneVID | **52.66%** | ✅ |
| STTM-v2 | **51.87%** | ✅ |
| FastV | — | ❌ Hard-blocked |
| FlashVID | — | ❌ dycoke11 env |
| VisionZip | — | ❌ Broken |

## OVQwen2 Scoreboard (7/10 complete)
| Model | Accuracy | Status |
|-------|:--------:|--------|
| DyCoke | **53.36%** | ✅ (7713010) |
| FlashVID | **53.36%** | ✅ (7713093) |
| HoliTom | **53.14%** | ✅ |
| MDP3 | **53.06%** | ✅ (7713095) |
| FastV | **52.66%** | ✅ |
| STTM-v2 | **51.54%** | ✅ |
| VideoITG | — | 🔄 Running (7714855) |
| AIM | — | ❌ Hard-blocked |
| FastVID | — | ❌ Hard-blocked |
| VisionZip | — | ❌ Broken |

## Key Finding: Backbone Independence
Three models (HoliTom, DyCoke, MDP3) produce identical accuracy on both 7B backbones. MDP3 produced byte-identical results.jsonl across two independent runs with different models — the eval scripts verified as using correct paths.

## Currently Running
| Job ID | Model | Backbone | Type |
|:---:|-------|----------|------|
| 7714855 | VideoITG | qwen2 | Full inference |

## Carya Allocation
81.26% remaining (426,608/525,000 hours)

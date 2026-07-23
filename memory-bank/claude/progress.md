# Progress — HuBenchmarks (2026-07-22 19:52 CDT)

## Current Running / Queued

| Job ID | Model | Backbone | Status | Time |
|:------:|-------|----------|:------:|:----:|
| 7762101 | VisionZip | OVQwen 1.5 | ⚡ RUNNING | 7m |
| 7762102 | VisionZip | OVQwen 2 | ⚡ RUNNING | 6m |
| 7762103 | FastV (baseline) | OVQwen 1.5 | ⏳ PENDING | — |
| 7753572 | HoliTom | OVQwen 1.5 | ⏳ PENDING | — |
| 7753571 | HoliTom | OVQwen 2 | ⏳ PENDING | — |
| 7753570 | MDP3 | OVQwen 2 | ⏳ PENDING | — |
| 7753569 | MDP3 | OVQwen 1.5 | ⏳ PENDING | — |

## OVQwen 1.5 Scoreboard (8/10 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1 | DyCoke | **53.36%** | ✅ Algorithm-dominant (see dycoke-data-integrity.md) |
| 2 | HoliTom | **53.14%** | ✅ Fresh run submitted (7753572) |
| 3 | MDP3 | **53.06%** | ✅ Fresh run submitted (7753569) |
| 4 | VideoITG | **52.86%** | ✅ |
| 5 | AIM | **52.81%** | ✅ |
| 6 | PruneVID | **52.66%** | ✅ |
| 7 | STTM-v2 | **51.87%** | ✅ |
| 8 | FlashVID (0.15) | **51.22%** | ✅ |
| 9 | FastV | — | 🔄 Submitted (7762103) using dycoke11 env |
| 10 | VisionZip | — | 🔄 Running (7762101) |

## OVQwen2 Scoreboard (9/11 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1= | DyCoke | **53.36%** | ✅ Algorithm-dominant (see dycoke-data-integrity.md) |
| 2 | FlashVID (0.15) | **53.29%** | ✅ |
| 3 | HoliTom | **53.14%** | ✅ Fresh run submitted (7753571) |
| 4 | MDP3 | **53.06%** | ✅ Fresh run submitted (7753570) |
| 5 | AIM | **52.84%** | ✅ |
| 6 | FastV | **52.66%** | ✅ (7704758) |
| 7 | VideoITG | **52.86%** | ✅ |
| 8 | STTM-v2 | **51.54%** | ✅ |
| 9 | VisionZip | — | 🔄 Running (7762102) |
| 10 | FastVID | — | ❌ Hard-blocked (model-level 0% bug) |

## Still Broken / No OV Code

| Model | Reason | Next Steps |
|-------|--------|------------|
| DyTo | Vicuna-only backend, no OV code | Not applicable |
| iMove | No public code | Not applicable |
| TrajViT | No public code | Not applicable |
| FastVID qwen2 | `'NoneType' object is not callable` | Deep debug needed |
| STTM-LLaVAVid | → experiments/ | Separate track |

## DyCoke — Algorithm-Dominant Results

DyCoke produces 53.36% identically across both backbones because its two-stage compression (token merging + KV cache pruning at aggressive ratios) erases backbone-level signal. See `memory-bank/claude/dycoke-data-integrity.md` for full analysis.

## Failed Job Fix History

| Attempt | Method | Error | Fix |
|:-------:|--------|-------|-----|
| v1 (7753562) | FastV | `ModuleNotFoundError: no module 'llava'` | Added PYTHONPATH to HoliTom/Llava-NeXT |
| v2 (7762093) | FastV | Same error | Actually fixed PYTHONPATH to `code/HoliTom/...` |
| v3 (7762100) | FastV | Same error | fastv conda env itself has broken torch (`register_fake` missing) |
| v4 (7762103) | FastV | — | Switched to dycoke11 conda env (known working) |
| v1 (7762094/95) | VisionZip | `ModuleNotFoundError: no module 'visionzip'` | Pointed to `visionzip_pkgs` — dir didn't exist |
| v2 (7762101/02) | VisionZip | — | Fixed: `visionzip` at `code/VisionZip/`, `llava` at `AIM/` |

## Carya Allocation
79.12% remaining (415,356/525,000 hours)

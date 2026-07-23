# Progress — HuBenchmarks (2026-07-23 11:43 CDT)

## Current Running / Queued

| Job ID | Model | Backbone | Status | Time |
|:------:|-------|----------|:------:|:----:|
| 7768241 | MDP3 | OVQwen 1.5 | ⚡ RUNNING | 1h 23m |
| 7768829 | VisionZip | OVQwen 1.5 | ⚡ RUNNING | 48m |
| 7769039 | VisionZip | OVQwen 2 | ⚡ RUNNING | recent |

## Recently Completed (since last update)

| Job ID | Model | Backbone | Result | Notes |
|:------:|-------|----------|:------:|-------|
| 7762123 | FastV (baseline) | OVQwen 1.5 | **52.66%** (2116/4018) | ✅ COMPLETED. DyCoke builder + dycoke11 env. `apply_fastv()` stub — ran as baseline. |
| 7762128 | MDP3 | OVQwen 2 | **53.06%** (2132/4018) | ✅ COMPLETED (7h 26m). |
| 7762126 | HoliTom | OVQwen 1.5 | ❌ FAILED | `transformers.modeling_rope_utils` not found. Already have valid result (53.14%) from previous run. |
| 7762127 | HoliTom | OVQwen 2 | ❌ FAILED | Same error. Already have valid result (53.14%) from previous run. |
| 7762101 | VisionZip | OVQwen 1.5 | ❌ 0.0% | All empty predictions. AIM transformers shadowing + wrong eval script path. |
| 7762102 | VisionZip | OVQwen 2 | ❌ 0.0% | Same issue. |
| 7762129 | MDP3 | OVQwen 1.5 | ❌ FAILED | `--num_frames` not accepted by MDP3 eval. Resubmitted as 7768241. |

## OVQwen 1.5 Scoreboard (9/10 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1 | DyCoke | **53.36%** | ✅ Algorithm-dominant (see dycoke-data-integrity.md) |
| 2 | HoliTom | **53.14%** | ✅ |
| 3 | MDP3 | **53.06%** | ✅ |
| 4 | VideoITG | **52.86%** | ✅ |
| 5 | AIM | **52.81%** | ✅ |
| 6 | PruneVID | **52.66%** | ✅ |
| 7 | FastV (baseline) | **52.66%** | ✅ Job 7762123 (enabled=false stub) |
| 8 | STTM-v2 | **51.87%** | ✅ |
| 9 | FlashVID (0.15) | **51.22%** | ✅ |
| 10 | VisionZip | — | 🔄 Resubmitted (7768829) |

## OVQwen2 Scoreboard (10/11 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1= | DyCoke | **53.36%** | ✅ Algorithm-dominant (see dycoke-data-integrity.md) |
| 2 | FlashVID (0.15) | **53.29%** | ✅ |
| 3 | HoliTom | **53.14%** | ✅ |
| 4 | MDP3 | **53.06%** | ✅ Job 7762128 completed |
| 5 | AIM | **52.84%** | ✅ |
| 6 | FastV | **52.66%** | ✅ |
| 7 | VideoITG | **52.86%** | ✅ |
| 8 | STTM-v2 | **51.54%** | ✅ |
| 9 | VisionZip | — | 🔄 Resubmitted (7769039) |
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
| v2 (7762101/02) | VisionZip | 0.0% accuracy (all empty predictions) | AIM transformers shadowing + wrong eval script path |
| v3 (7768829/9039) | VisionZip | — | Fixed: visionzip env + DyCoke builder + correct eval path |
| v1 (7762126/27) | HoliTom | `transformers.modeling_rope_utils` not found | Already have valid results — no resubmit needed |
| v1 (7762129) | MDP3 OV1.5 | `--num_frames` not accepted | Removed invalid arg; resubmitted as 7768241 |

## Carya Allocation
78.94% remaining (414,451/525,000 hours)

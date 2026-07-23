# Progress — HuBenchmarks (2026-07-23 12:25 CDT)

## Current Running / Queued

| Job ID | Model | Status | Time |
|:------:|-------|:------:|:----:|
| 7768241 | MDP3 | ⚡ RUNNING | ~2h |
| 7768829 | VisionZip | ⚡ RUNNING | ~1h |
| 7769039 | VisionZip | ⚡ RUNNING | ~45m |

## Scoreboard (12/13 complete, 1 VisionZip pending)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1= | **DyCoke** | **53.36%** | ✅ |
| 1= | **FlashVID** (0.25) | **53.36%** | ✅ |
| 3 | **FlashVID** (0.15) | **53.29%** | ✅ |
| 4 | **HoliTom** | **53.14%** | ✅ |
| 5 | **MDP3** | **53.06%** | ✅ |
| 6 | **VideoITG** | **52.86%** | ✅ |
| 7 | **AIM** | **52.84%** | ✅ |
| 8= | **PruneVID** | **52.66%** | ✅ |
| 8= | **FastV** (baseline) | **52.66%** | ✅ Job 7762123 |
| 10 | **STTM-v2** | **51.87%** | ✅ |
| 11 | **FlashVID** (qwen15 templ) | **51.22%** | ✅ |
| 12 | **VideoITG** (simplified) | **34.89%** | ✅ |
| 13 | **VisionZip** | — | 🔄 Resubmitted (7768829, 7769039) |

## Critical Correction (2026-07-23)

The project previously tracked "Qwen1.5 backbone" and "Qwen2 backbone" as separate columns. This was a **misconception** discovered today:

- `lmms-lab/llava-ov-7b` **already uses Qwen2 internally** — architecture is `LlavaQwenForCausalLM`, hidden_size=3584, training path includes `Qwen2-7B-Instruct`
- The duplicate `llava-ov-7b-qwen2` weight directory was a byte-for-byte copy. Deleted on Carya with README explanation.
- The `qwen_1_5` vs `qwen_2` conv templates only control **prompt formatting**, not model weights
- All "identical across backbones" findings (DyCoke, HoliTom, MDP3) are therefore expected — same model, different prompt format
- Results listed under both columns are the **same 12 methods**, not 24. The table has been consolidated.

## Still Broken / No OV Code

| Model | Reason | Next Steps |
|-------|--------|------------|
| DyTo | Vicuna-only backend, no OV code | Not applicable |
| iMove | No public code | Not applicable |
| TrajViT | No public code | Not applicable |
| FastVID qwen2 | `'NoneType' object is not callable` | Deep debug needed |
| STTM-LLaVAVid | → experiments/ | Separate track |

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
~78.9% remaining (414,430/525,000 hours)

# Progress — HuBenchmarks (2026-07-23 13:56 CDT)

## Current Running / Queued

| Job ID | Model | Stage | Status | Time |
|:------:|-------|:-----:|:------:|:----:|
| 7768241 | MDP3 | Stage 1 (OV1.5 rerun) | ⚡ RUNNING | ~3h |
| 7768829 | VisionZip | Stage 1 | ⚡ RUNNING | ~2.5h |
| 7769192 | stage2_aim | Stage 2 | ⚡ RUNNING | ~20m |
| 7769194 | stage2_fastv | Stage 2 | ⚡ RUNNING | ~20m |
| 7769212 | **DyTo** (vicuna) | Other-backbone | ⏳ PENDING | — |
| 7769193 | stage2_dycoke | Stage 2 | ⏳ PENDING | — |
| 7769195 | stage2_flashvid | Stage 2 | ⏳ PENDING | — |
| 7769196 | stage2_holitom | Stage 2 | ⏳ PENDING | — |
| 7769197 | stage2_mdp3 | Stage 2 | ⏳ PENDING | — |
| 7769198 | stage2_videoitg | Stage 2 | ⏳ PENDING | — |
| 7769201 | VisionZip v3 | Stage 1 | ⏳ PENDING | — |
| 7769209 | stage2_sttm | Stage 2 | ⏳ PENDING | — |

## Scoreboard — Stage 1 (LLaVA-OV-7B, 12/13 complete)

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
| — | ~~FastV~~ → **Backbone baseline** (stub, `enabled:false`) | **52.66%** | ⚠️ NOT a method |
| 10 | **STTM-v2** | **51.87%** | ✅ |
| 11 | **FlashVID** (qwen15 templ) | **51.22%** | ✅ |
| 12 | **VideoITG** (simplified) | **34.89%** | ✅ |
| 13 | **VisionZip** | — | 🔄 v3=7769201, v1=7768829 |

## Stage 2 Scoreboard (LLaVA-Video-7B) — ALL PENDING

| Method | Status | Job ID |
|--------|:------:|:------:|
| AIM | ⚡ RUNNING | 7769192 |
| FastV | ⚡ RUNNING | 7769194 |
| DyCoke | ⏳ PENDING | 7769193 |
| FlashVID | ⏳ PENDING | 7769195 |
| HoliTom | ⏳ PENDING | 7769196 |
| MDP3 | ⏳ PENDING | 7769197 |
| VideoITG | ⏳ PENDING | 7769198 |
| STTM | ⏳ PENDING | 7769209 |

*Not applicable to Stage 2: FastVID (model bug), PruneVID (PLLaVA), VisionZip (LLaVA-1.5), DyTo (Vicuna)*

## Other-backbone Scoreboard

| Method | Backbone | Accuracy | Job ID |
|--------|----------|:--------:|:------:|
| DyTo | LLaVA-NeXT Vicuna-7B | — | 🔄 7769212 (patched builder) |
| PruneVID | PLLaVA-7B | **52.66%** | ✅ Has own sbatch |
| iMove | LLaVA-NeXT | — | ❌ No public code |
| TrajViT | LLaVA-NeXT | — | ❌ No public code |

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
| DyTo | Vicuna-only backend, no OV code | Builder patched (LlavaLlamaForCausalLM import). ⏳ 7769212 |
| iMove | No public code | Not applicable |
| TrajViT | No public code | Not applicable |
| FastVID | `'NoneType' object is not callable` | Deep debug needed |

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
| v3b (7769201) | VisionZip | v2 had flash_attn issue | Using full eval_visionzip.py with sdpa attn_implementation |
| v1 (7762126/27) | HoliTom | `transformers.modeling_rope_utils` not found | Already have valid results — no resubmit needed |
| v1 (7762129) | MDP3 OV1.5 | `--num_frames` not accepted | Removed invalid arg; resubmitted as 7768241 |
| v1-3 (multiple) | DyTo | `NameError: LlavaLlamaForCausalLM` | Added explicit import in DYTO/llava/model/builder.py. Resubmitted 7769212 |

## Carya Allocation
~78.9% remaining (414,274/525,000 hours)

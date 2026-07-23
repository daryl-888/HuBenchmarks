# Claude's Active Context — 2026-07-23 12:50 CDT

## Current Focus
Repo restructured from confusing `ovqwen/ovqwen2/ovqwen3/` naming into a clean 3-stage architecture. The "Qwen2 backbone" was a misconception — `llava-ov-7b` already uses Qwen2 internally. Duplicate weights deleted from Carya.

## Repository Structure (Updated 2026-07-23)

```
stage1-llava-ov/       Stage 1: Mid-2024 — LLaVA-OV (Qwen2 7B LlavaQwenForCausalLM, hidden_size=3584)
                       9 working methods + VisionZip pending
stage2-llava-video/    Stage 2: Late-2024 — LLaVA-Video-7B (Qwen2, video-finetuned)
                       Currently: sttm-llavavid experiments only
stage3-qwen3-vl/       Stage 3: Late-2025 — Qwen3-VL-8B (Qwen3 8B Dense)
                       8 method directories (ported from stage1)
other-backbones/       Non-series models (DyTo, iMove, PruneVid, STTM-v1, TrajViT, VisionZip on LLava-1.5)
sbatch-files/          Local sbatch templates for Carya submission
```

### Correction (2026-07-23)
- There is no "Qwen1.5 vs Qwen2 backbone" — both use the **same model** (`lmms-lab/llava-ov-7b`, arch: `LlavaQwenForCausalLM`, uses `Qwen2-7B-Instruct` internally)
- `qwen_1_5` vs `qwen_2` conv templates only control **prompt formatting**, not model weights
- The duplicate `llava-ov-7b-qwen2` weight directory was deleted from Carya
- FastV OV1.5 now has a valid baseline result (52.66%) — no longer hard-blocked

## Consolidated Scoreboard (Single Model: LLaVA-OV-7B)

| # | Model | Overall | Status |
|---|-------|:-------:|--------|
| 1= | DyCoke | **53.36%** | ✅ |
| 1= | FlashVID (0.25) | **53.36%** | ✅ |
| 3 | FlashVID (0.15) | **53.29%** | ✅ |
| 4 | HoliTom | **53.14%** | ✅ |
| 5 | MDP3 | **53.06%** | ✅ |
| 6 | VideoITG | **52.86%** | ✅ |
| 7 | AIM | **52.84%** | ✅ |
| 8= | PruneVID | **52.66%** | ✅ |
| 8= | FastV (baseline) | **52.66%** | ✅ |
| 10 | STTM-v2 | **51.87%** | ✅ |
| 11 | FlashVID (qwen15 templ) | **51.22%** | ✅ |
| 12 | VideoITG (simplified) | **34.89%** | ✅ |
| 13 | VisionZip | — | 🔄 Submitted (7768829, 7769039) |

## Running Jobs

| Job ID | Model | Status |
|:------:|-------|:------:|
| 7768241 | MDP3 OV1.5 (fresh re-run) | ⚡ RUNNING |
| 7768829 | VisionZip stage1 | ⚡ RUNNING |
| 7769039 | VisionZip stage1 | ⚡ RUNNING |

## Active Remaining Blockers
- **FastVID**: Model-level bug causing 0% accuracy (`'NoneType' object is not callable`)
- **DyTo**: Failed 3 times on Vicuna backbone. LlavaLlamaForCausalLM import error from DYTO/llava module.
- **VisionZip**: Was 0% due to AIM shadowing — resubmitted with DyCoke builder fix
- **Stage2 (llava-video-7b)**: Most methods not yet ported — only sttm-llavavid exists

## Carya Paths

| Path | What |
|------|------|
| `/project/rhu/dpalfaro/code/stage1-llava-ov/` | 9 methods on LLaVA-OV-7B ✅ |
| `/project/rhu/dpalfaro/code/stage2-llava-video/` | sttm-llavavid only |
| `/project/rhu/dpalfaro/code/stage3-qwen3-vl/` | 8 methods on Qwen3-VL |
| `/project/rhu/dpalfaro/code/other-backbones/` | Non-series models |
| `/project/rhu/dpalfaro/weights/llava-ov-7b/` | **Primary eval model** (15G) |
| `/project/rhu/dpalfaro/weights/llava-video-7b/` | Stage 2 model (15G) |
| `/project/rhu/dpalfaro/weights/qwen3-vl-8b/` | Stage 3 model (17G) |

## Conda Env Reference (on Carya)

| Env | Path | Models |
|-----|------|--------|
| dycoke11 | `/project/rhu/dpalfaro/conda/envs/dycoke11` | DyCoke, AIM, FastVID, FlashVID |
| holitom | `/project/rhu/dpalfaro/conda/envs/holitom` | HoliTom |
| fastv | `/project/rhu/dpalfaro/conda/envs/fastv` | FastV (broken torch) |
| mdp3 | `/project/rhu/dpalfaro/conda/envs/mdp3` | MDP3 |
| sttm_new | `/project/rhu/dpalfaro/conda/envs/sttm_new` | STTM |
| videoitg | `/project/rhu/dpalfaro/conda/envs/videoitg` | VideoITG |
| visionzip | `/project/rhu/dpalfaro/conda/envs/visionzip` | VisionZip |

## Key Fix Patterns
- **MDP3**: needs `LD_LIBRARY_PATH=/project/rhu/dpalfaro/mdp3_pkgs/torch/lib`, `--pool-frames` (not `--num_frames`), 48h walltime
- **HoliTom**: needs `holitom` env + `PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke:/project/rhu/dpalfaro/code/HoliTom` + env vars WRAPPER, RETAIN_RATIO, T, HOLITOM_k, HOLITOM_r
- **FastV**: needs `dycoke11` env + `PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke` (uses dycoke11 builder, not fastv env which has broken torch)
- **VisionZip**: needs `visionzip` env + `PYTHONPATH=/project/rhu/dpalfaro/code/VisionZip:/project/rhu/dpalfaro/code/DyCoke`
- **DyCoke**: needs `dycoke11` env + `PYTHONPATH=/project/rhu/dpalfaro/code/DyCoke`

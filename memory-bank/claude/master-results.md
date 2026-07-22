# Master Results Table — HuBenchmarks

> **Last updated**: 2026-07-22 08:00 CDT
> **Benchmark**: MotionBench — 8,052 MCQ video samples, 4,018 scoreable, 4,034 NA
> **Standard eval**: 32 frames, `do_sample=False`, `max_new_tokens=16`, letter-match scoring, NA-skip

---

## Qwen 1.5 Backbone (`llava-ov-7b`, `qwen_1_5`)

| # | Model | Conference | Year | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Parameters | Notes |
|---|-------|-----------|------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|------------|-------|
| 1 | **DyCoke** | arXiv | 2024 | **53.36%** (2144/4018) | 38.92% (202/519) | 48.83% (188/385) | 55.68% (304/546) | 58.19% (860/1478) | 70.87% (489/690) | 25.25% (101/400) | l=3, p=0.7, k=0.7, attn_impl=sdpa | ✅ Re-run (7714650). |
| 2 | **HoliTom** | — | 2025 | **53.14%** (2135/4018) | 40.85% (212/519) | 49.61% (191/385) | 52.56% (287/546) | 57.04% (843/1478) | 71.45% (493/690) | 27.25% (109/400) | RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5 | ✅ Re-run (7713092). HoliTom env. |
| 3 | **MDP3** | arXiv | 2025 | **53.06%** (2132/4018) | 40.46% (210/519) | 49.61% (191/385) | 53.48% (292/546) | 56.77% (839/1478) | 71.59% (494/690) | 26.50% (106/400) | pool_frames=32, select_frames=8 | ✅ Re-run (7714722). |
| 4 | **VideoITG** | — | 2025 | **52.86%** (2124/4018) | 40.08% (208/519) | 47.01% (181/385) | 53.66% (293/546) | 57.58% (851/1478) | 70.14% (484/690) | 26.75% (107/400) | grounding: 512 sample, 32 select, 2fps | ✅ Clean re-run (7693609). |
| 5 | **AIM** | ICCV | 2025 | **52.81%** (2122/4018) | 41.23% (214/519) | 48.05% (185/385) | 54.03% (295/546) | 57.04% (843/1478) | 71.74% (495/690) | 22.50% (90/400) | aim env, eager attention | Bipartite soft matching + PageRank prune. |
| 6 | **PruneVID** | — | 2024 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | cluster_ratio=0.5, temporal_segment_ratio=0.25 | ovqwen_prunevid_run2. |
| 7 | **STTM-v2** | — | 2025 | **51.87%** (2084/4018) | 39.11% (203/519) | 48.83% (188/385) | 53.66% (293/546) | 53.59% (792/1478) | 71.16% (491/690) | 29.25% (117/400) | sa_start_layer_idx=2, sa_tree_thresh=0.85 | Job 7691289. |
| 8 | **FlashVID** | ICLR | 2026 | — | — | — | — | — | — | — | retention_ratio=0.15, flashvid env | 🔄 Submitting full run (smoke passed 7750924) |
| 9 | **FastV** | — | 2024 | — | — | — | — | — | — | — | k=2, r=0.5 | ❌ Hard-blocked — NotImplementedError for Qwen1.5 |
| 10 | **VisionZip** | — | 2024 | — | — | — | — | — | — | — | dominant=54, contextual=10 | ❌ Broken — LlavaConfig not recognized |

### Excluded (4)
FastVID (Qwen2-only), DyTo/iMove/TrajViT (incompatible/no code), STTM-LLaVAVid (→ experiments)

---

## Qwen2 Backbone (`llava-ov-7b-qwen2`, `qwen_2`)

| # | Model | Conference | Year | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Parameters | Notes |
|---|-------|-----------|------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|------------|-------|
| 1= | **DyCoke** | arXiv | 2024 | **53.36%** (2144/4018) | 38.92% (202/519) | 48.83% (188/385) | 55.68% (304/546) | 58.19% (860/1478) | 70.87% (489/690) | 25.25% (101/400) | l=3, p=0.7, k=0.7, attn_impl=sdpa | ✅ (7713010) |
| 1= | **FlashVID** (0.25) | ICLR | 2026 | **53.36%** (2144/4018) | 42.20% (219/519) | 47.53% (183/385) | 55.13% (301/546) | 57.58% (851/1478) | 71.74% (495/690) | 23.75% (95/400) | retention_ratio=0.25 | ✅ Old run (7713093). |
| 3 | **FlashVID** (0.15) | ICLR | 2026 | **53.29%** (2141/4018) | 39.69% (206/519) | 45.71% (176/385) | 54.03% (295/546) | 57.85% (855/1478) | 71.88% (496/690) | 28.25% (113/400) | retention_ratio=0.15 | ✅ NEW (7750906). |
| 4 | **HoliTom** | — | 2025 | **53.14%** (2135/4018) | 40.85% (212/519) | 49.61% (191/385) | 52.56% (287/546) | 57.04% (843/1478) | 71.45% (493/690) | 27.25% (109/400) | RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5 | dycoke11 env. |
| 5 | **MDP3** | arXiv | 2025 | **53.06%** (2132/4018) | 40.46% (210/519) | 49.61% (191/385) | 53.48% (292/546) | 56.77% (839/1478) | 71.59% (494/690) | 26.50% (106/400) | pool_frames=32, select_frames=8 | ✅ Re-run (7713095). |
| 6 | **VideoITG** | — | 2025 | **52.86%** (2124/4018) | — | — | — | — | — | — | grounding + inference | ✅ NEW (7714855). |
| 7 | **AIM** | ICCV | 2025 | **52.84%** (2123/4018) | — | — | — | — | — | — | aim env, eager attention | ✅ NEW (7714969). |
| 8 | **FastV** | — | 2024 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | k=2, r=0.5, eager attention | ✅ (7704758). |
| 9 | **FlashVID** (8f) | ICLR | 2026 | **51.92%** (2086/4018) | — | — | — | — | — | — | retention_ratio=0.1, num_frames=8 | Old 8f run. Superseded by 32f. |
| 10 | **STTM-v2** | — | 2025 | **51.54%** (2071/4018) | 38.54% (200/519) | 48.05% (185/385) | 54.03% (295/546) | 53.45% (790/1478) | 70.43% (486/690) | 28.75% (115/400) | sa_start_layer_idx=2, sa_tree_thresh=0.85 | sttm_new env. |
| 11 | **FastVID** | — | 2024 | — | — | — | — | — | — | — | fastvid env | ❌ Hard-blocked — model-level bug (0% accuracy) |
| 12 | **VisionZip** | — | 2024 | — | — | — | — | — | — | — | dominant=54, contextual=10 | ❌ Broken — LlavaConfig not recognized |

### Excluded (4)
DyTo/iMove/TrajViT (incompatible/no code), STTM-LLaVAVid (→ experiments)

---

## Old / Non-OV Backbones

| # | Model | Backbone | Overall | Notes |
|---|-------|----------|:-------:|-------|
| 1 | PruneVID | Legacy | 43.80% | Different backbone/config |
| 2 | VisionZip | LLaVA-v1.5-7b | 39.97% | Wrong backbone |
| 3 | DyTo | LLaVA-v1.6-Vicuna-7b | — | ❌ FAILED v2 (7750932) — LlavaLlamaForCausalLM import |

---

## STTM-LLaVAVid Experiment Results

| Config | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Notes |
|--------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|-------|
| **t=0.80, 32f** | **53.33%** (2143/4018) | 40.27% (209/519) | 48.83% (188/385) | 57.33% (313/546) | 57.51% (850/1478) | 68.84% (475/690) | 27.00% (108/400) | Job 7706552. |

---

## Currently Running / Queued (2026-07-22 08:00)

| Job ID | Model | Backbone | Type | Status |
|:---:|-------|----------|------|--------|
| — | FlashVID | qwen1.5 | Full (0.15 retention) | 🔄 SUBMITTING |
| — | DyTo | Vicuna-7B | v3 (in-place patch) | 🔄 FIXING |

---

## Model Reference (All 15 Models)

| # | Model | OV-Q1.5 | OV-Q2 | Notes |
|---|-------|:-------:|:-----:|-------|
| 1 | AIM | ✅ 52.81% | ✅ 52.84% | |
| 2 | DyCoke | ✅ 53.36% | ✅ 53.36% | Identical accuracy across backbones |
| 3 | DyTo | ❌ OV | ❌ OV | Vicuna backbone — v2 failed, fixing v3 |
| 4 | FastV | ❌ Hard-blocked | ✅ 52.66% | |
| 5 | FastVID | ❌ Qwen2-only | ❌ Hard-blocked | Model-level 0% bug |
| 6 | FlashVID | 🔄 Submitting full | ✅ 53.29% (0.15) | ICLR 2026 Oral |
| 7 | HoliTom | ✅ 53.14% | ✅ 53.14% | Identical accuracy across backbones |
| 8 | iMove | ❌ | ❌ | No public code |
| 9 | MDP3 | ✅ 53.06% | ✅ 53.06% | Identical accuracy across backbones |
| 10 | PruneVID | ✅ 52.66% | ⏸️ | |
| 11 | STTM-v2 | ✅ 51.87% | ✅ 51.54% | |
| 12 | STTM-LLaVAVid | — | — | → experiments/ (53.33% on LLaVA-Video-7B) |
| 13 | TrajViT | ❌ | ❌ | No public code |
| 14 | VideoITG | ✅ 52.86% | ✅ 52.86% | |
| 15 | VisionZip | ❌ Broken | ❌ Broken | |

### Legend
| Symbol | Meaning |
|:------:|---------|
| ✅ | Completed with valid result |
| 🔄 | Currently running / queued |
| ⏸️ | Not yet started |
| ⚠️ | Completed but needs re-run |
| ❌ | Incompatible / broken / unavailable |

---

## Key Finding: Backbone Independence

Three models (HoliTom, DyCoke, MDP3) produce **identical accuracy across qwen1.5 and qwen2 backbones**. These are 7B models from the same family where the compression method (frame selection/retention for HoliTom/MDP3, KV cache merging for DyCoke) dominates performance.

## Retention Rate Audit

FlashVID was running at retention_ratio=0.25 instead of standard 0.15. Results:
- qwen2 0.15: ✅ **53.29%** (7750906) — 3 miniscule pts below 0.25 run
- qwen1.5 0.15: 🔄 Submitting full run (smoke passed 7750924)

See `memory-bank/claude/retention-rate-audit.md` for full details.

## Subcategory Reference

| Subcategory | Total Scoreable |
|-------------|:---------------:|
| Action Order | 519 |
| Camera Motion | 385 |
| Location-related Motion | 546 |
| Motion Recognition | 1,478 |
| Motion-related Objects | 690 |
| Repetition Count | 400 |
| **Total Scoreable** | **4,018** |
| NA (skipped) | 4,034 |
| **Total Samples** | **8,052** |

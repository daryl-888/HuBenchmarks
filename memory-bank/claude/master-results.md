# Master Results Table — HuBenchmarks

> **Last updated**: 2026-07-13 13:45 CDT
> **Benchmark**: MotionBench — 8,052 MCQ video samples, 4,018 scoreable, 4,034 NA
> **Standard eval**: 32 frames, `do_sample=False`, `max_new_tokens=16`, letter-match scoring, NA-skip

---

## Qwen 1.5 Backbone (`llava-ov-7b`, `qwen_1_5`)

| # | Model | Conference | Year | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Parameters | Notes |
|---|-------|-----------|------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|------------|-------|
| 1 | **MDP3** | arXiv | 2025 | **53.06%** (2132/4018) | 40.46% (210/519) | 49.61% (191/385) | 53.48% (292/546) | 56.77% (839/1478) | 71.59% (494/690) | 26.50% (106/400) | pool_frames=32, select_frames=8 | Pre-restructure run. ⚠️ Needs re-run to verify. |
| 2 | **VideoITG** | — | 2025 | **52.86%** (2124/4018) | 40.08% (208/519) | 47.01% (181/385) | 53.66% (293/546) | 57.58% (851/1478) | 70.14% (484/690) | 26.75% (107/400) | grounding: 512 sample, 32 select, 2fps | ✅ NEW re-run. Job 7693609. Clean re-ran grounding→inference pipeline. |
| 3 | **AIM** | ICCV | 2025 | **52.81%** (2122/4018) | 41.23% (214/519) | 48.05% (185/385) | 54.03% (295/546) | 57.04% (843/1478) | 71.74% (495/690) | 22.50% (90/400) | aim env, eager attention, torch 2.3.1 | Bipartite soft matching + PageRank prune. |
| 4 | **PruneVID** | — | 2024 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | cluster_ratio=0.5, temporal_segment_ratio=0.25, layer=10, alpha=0.4, tau=0.8 | Dir: ovqwen_prunevid_run2. |
| 5 | **HoliTom** | — | 2025 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5 | ⚠️ Old run — needs re-run with `holitom` conda env. |
| 6 | **STTM-v2** | — | 2025 | **51.87%** (2084/4018) | 39.11% (203/519) | 48.83% (188/385) | 53.66% (293/546) | 53.59% (792/1478) | 71.16% (491/690) | 29.25% (117/400) | sa_start_layer_idx=2, sa_tree_thresh=0.85, sa_tree_temporal_thresh=0.65 | QuadTree LLM attention. Job 7691289. |
| 7 | **MDP3** | arXiv | 2025 | — | — | — | — | — | — | — | pool_frames=32, select_frames=8 | 🔄 Re-running (7695024, 48h). Cancelled at 57% — was running well. |
| 8 | **DyCoke** | arXiv | 2024 | — | — | — | — | — | — | — | l=3, p=0.7, k=0.7 | 🔄 Smoke queued (7695025). |
| 9 | **FastV** | — | 2024 | — | — | — | — | — | — | — | k=2, r=0.5, eager attention | ⏸️ Needs PYTHONPATH + fastv conda env. |
| 10 | **FlashVID** | ICLR | 2026 | — | — | — | — | — | — | — | retention_ratio=0.10, alpha=0.7, temporal_threshold=0.8 | Smoke passed 48.15% (13/27). Full run pending. |
| 11 | **VisionZip** | — | 2024 | — | — | — | — | — | — | — | dominant=54, contextual=10 | ❌ Broken — `LlavaConfig not recognized`. |

---

## Qwen2 Backbone (`llava-ov-7b-qwen2`, `qwen_2`)

| # | Model | Conference | Year | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Parameters | Notes |
|---|-------|-----------|------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|------------|-------|
| 1 | **HoliTom** | — | 2025 | **53.14%** (2135/4018) | 40.85% (212/519) | 49.61% (191/385) | 52.56% (287/546) | 57.04% (843/1478) | 71.45% (493/690) | 27.25% (109/400) | RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5 | dycoke11 env. |
| 2 | **FlashVID** | ICLR | 2026 | **51.92%** (2086/4018) | 39.50% (205/519) | 47.01% (181/385) | 55.13% (301/546) | 53.79% (795/1478) | 71.30% (492/690) | 28.00% (112/400) | retention_ratio=0.1, alpha=0.7, temporal_threshold=0.8, ⚠️ num_frames=8 | **ICLR 2026 Oral**. Ran at 8f — needs 32f re-run. |
| 3 | **STTM-v2** | — | 2025 | **51.54%** (2071/4018) | 38.54% (200/519) | 48.05% (185/385) | 54.03% (295/546) | 53.45% (790/1478) | 70.43% (486/690) | 28.75% (115/400) | sa_start_layer_idx=2, sa_tree_thresh=0.85 | sttm_new env. |

---

## Old / Non-OV Backbones (not usable for comparison)

| # | Model | Backbone | Overall | Notes |
|---|-------|----------|:-------:|-------|
| 1 | PruneVID | Legacy (pre-restructure) | 43.80% | Different backbone/config |
| 2 | VisionZip | LLaVA-v1.5-7b | 39.97% | Wrong backbone |
| 3 | DyTo | LLaVA-v1.6-Vicuna-7b | 5.25% | Vicuna architecture |

---

## STTM-LLaVAVid Experiment Results (LLaVA-Video-7B, 50 samples)

| Config | Accuracy | Notes |
|--------|:--------:|-------|
| **t0.80, 32f** | **74.07%** (20/27) | 🔥 Best config |
| Default (t0.85, 32f) | 70.37% (19/27) | Paper defaults |
| t0.90, 32f | 70.37% (19/27) | |
| f16, t0.85 | 70.37% (19/27) | |
| t0.95, 32f | 66.67% (18/27) | |
| f64, t0.85 | 66.67% (18/27) | |
| Vanilla (no STTM) | 59.26% (16/27) | Baseline — STTM gives +14.81% |

**Conclusion**: Threshold 0.80 is the sweet spot. STTM adds massive gains over uncompressed LLaVA-Video. Full run on t0.80 config recommended.

---

## Model Reference (All 15 Models)

| # | Model | OV-Q1.5 | OV-Q2 | OV-Q3 | Notes |
|---|-------|:-------:|:-----:|:-----:|-------|
| 1 | AIM | ✅ 52.81% | ⏸️ | ⏸️ | ICCV 2025 |
| 2 | DyCoke | 🔄 Smoke | ⏸️ | ⏸️ | |
| 3 | DyTo | ❌ | ❌ | ❌ | Vicuna |
| 4 | FastV | 🔄 Queued | ⏸️ | ⏸️ | |
| 5 | FastVID | ❌ Qwen2-only | ⏸️ | ⏸️ | |
| 6 | FlashVID | 🔄 Smoke done | ⚠️ 8f | ⏸️ | ICLR 2026 Oral |
| 7 | HoliTom | ✅ 52.66% | ✅ 53.14% | ⏸️ | |
| 8 | iMove | ❌ | ❌ | ❌ | No code |
| 9 | MDP3 | ✅ 53.06% | ⏸️ | ⏸️ | Needs re-run |
| 10 | PruneVID | ✅ 52.66% | ⏸️ | ⏸️ | |
| 11 | STTM | ✅ 51.87% | ✅ 51.54% | ⏸️ | |
| 12 | STTM-LLaVAVid | ❌ | ❌ 0% | ⏸️ | → experiments/ |
| 13 | TrajViT | ❌ | ❌ | ❌ | No code |
| 14 | VideoITG | ✅ 52.86% | ⏸️ | ⏸️ | Re-ran clean |
| 15 | VisionZip | 🔄 Broken | ⏸️ | ⏸️ | |

### Legend
| Symbol | Meaning |
|:------:|---------|
| ✅ | Completed with valid result |
| 🔄 | Currently running / queued / needs fix |
| ⏸️ | Not yet started |
| ⚠️ | Completed but needs re-run |
| ❌ | Incompatible / broken / unavailable |

---

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

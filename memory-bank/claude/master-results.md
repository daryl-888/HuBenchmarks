# Master Results Table — HuBenchmarks

> **Last updated**: 2026-07-23 12:25 CDT
> **Benchmark**: MotionBench — 8,052 MCQ video samples, 4,018 scoreable, 4,034 NA
> **Standard eval**: 32 frames, `do_sample=False`, `max_new_tokens=16`, letter-match scoring, NA-skip

> **Critical Correction (2026-07-23)**: `llava-ov-7b` already uses Qwen2 internally (`LlavaQwenForCausalLM`, `hidden_size=3584`). The "qwen_1_5" vs "qwen_2" distinction is **only a conv template difference** (prompt formatting), not different model weights. The duplicate `llava-ov-7b-qwen2` directory on Carya has been deleted. Both columns in the table below run the **same model** with different templates — identical results are expected and correct.

---

## LLaVA-OV-7B (`llava-ov-7b`, `LlavaQwenForCausalLM`)

| # | Model | Conference | Year | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Parameters | Notes |
|---|-------|-----------|------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|------------|-------|
| 1 | **DyCoke** | arXiv | 2024 | **53.36%** (2144/4018) | 38.92% (202/519) | 48.83% (188/385) | 55.68% (304/546) | 58.19% (860/1478) | 70.87% (489/690) | 25.25% (101/400) | l=3, p=0.7, k=0.7 | Best overall. Two-stage token merging + KV cache prune. |
| 2 | **FlashVID** (0.25) | ICLR | 2026 | **53.36%** (2144/4018) | 42.20% (219/519) | 47.53% (183/385) | 55.13% (301/546) | 57.58% (851/1478) | 71.74% (495/690) | 23.75% (95/400) | retention_ratio=0.25 | ICLR 2026 Oral |
| 3 | **FlashVID** (0.15) | ICLR | 2026 | **53.29%** (2141/4018) | 39.69% (206/519) | 45.71% (176/385) | 54.03% (295/546) | 57.85% (855/1478) | 71.88% (496/690) | 28.25% (113/400) | retention_ratio=0.15 | Standard retention ratio |
| 4 | **HoliTom** | — | 2025 | **53.14%** (2135/4018) | 40.85% (212/519) | 49.61% (191/385) | 52.56% (287/546) | 57.04% (843/1478) | 71.45% (493/690) | 27.25% (109/400) | RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5 | |
| 5 | **MDP3** | arXiv | 2025 | **53.06%** (2132/4018) | 40.46% (210/519) | 49.61% (191/385) | 53.48% (292/546) | 56.77% (839/1478) | 71.59% (494/690) | 26.50% (106/400) | pool_frames=32, select_frames=8 | |
| 6 | **VideoITG** | — | 2025 | **52.86%** (2124/4018) | 40.08% (208/519) | 47.01% (181/385) | 53.66% (293/546) | 57.58% (851/1478) | 70.14% (484/690) | 26.75% (107/400) | grounding: 512 sample, 32 select, 2fps | |
| 7 | **AIM** | ICCV | 2025 | **52.84%** (2123/4018) | 41.43% (215/519) | 47.79% (184/385) | 54.03% (295/546) | 57.10% (844/1478) | 71.88% (496/690) | 22.25% (89/400) | aim env, eager attention | Bipartite soft matching + PageRank prune. |
| 8 | **PruneVID** | — | 2024 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | cluster_ratio=0.5, temporal_segment_ratio=0.25 | |
| 9 | **FastV** (baseline) | — | 2024 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | k=2, r=0.5 | apply_fastv() is a stub — ran as pure baseline |
| 10 | **STTM-v2** | — | 2025 | **51.87%** (2084/4018) | 39.11% (203/519) | 48.83% (188/385) | 53.66% (293/546) | 53.59% (792/1478) | 71.16% (491/690) | 29.25% (117/400) | sa_start_layer_idx=2, sa_tree_thresh=0.85 | |
| 11 | **FlashVID** (0.15, qwen15) | ICLR | 2026 | **51.22%** (2058/4018) | 39.50% (205/519) | 41.04% (158/385) | 54.95% (300/546) | 54.19% (801/1478) | 71.16% (491/690) | 25.75% (103/400) | retention_ratio=0.15, qwen_1_5 template | qwen_1_5 template run only. |
| 12 | **VideoITG** (simplified) | — | 2025 | **34.89%** (1402/4018) | 14.45% (75/519) | 45.19% (174/385) | 55.49% (303/546) | 44.65% (660/1478) | 20.29% (140/690) | 12.50% (50/400) | single-stage, no grounding | |
| 13 | **VisionZip** | — | 2024 | — | — | — | — | — | — | — | dominant=54, contextual=10 | 🔄 Rerunning (7768829) — was 0.0% due to AIM shadowing |

### Incomplete / Excluded (5)
- **FastVID**: ❌ Hard-blocked — model-level 0% bug (`'NoneType' object is not callable`)
- **DyTo**: ❌ Vicuna-only backbone, no OV code
- **iMove**: ❌ No public code
- **TrajViT**: ❌ No public code
- **STTM-LLaVAVid**: → experiments/ (separate track)

---

## Template Comparison: qwen_1_5 vs qwen_2

| Model | qwen_1_5 | qwen_2 | Delta | Notes |
|-------|:--------:|:------:|:-----:|-------|
| DyCoke | **53.36%** | **53.36%** | 0.00 | Identical — template has no effect |
| HoliTom | **53.14%** | **53.14%** | 0.00 | Identical |
| MDP3 | **53.06%** | **53.06%** | 0.00 | Identical |
| FastV (baseline) | **52.66%** | **52.66%** | 0.00 | Identical |
| FlashVID (0.15) | **51.22%** | **53.29%** | -2.07 | ⚠️ Different results! Only method with diff |

FlashVID's result difference (51.22% vs 53.29%) suggests one of its runs may have had different hyperparameters or a different code path — not just a template difference. Worth investigating.

### Key Correction (2026-07-23)
The project previously tracked "Qwen1.5 backbone" and "Qwen2 backbone" as separate columns. This was a misconception: `lmms-lab/llava-ov-7b` **already uses Qwen2 internally** (arch: `LlavaQwenForCausalLM`, hidden_size=3584, training path: `Qwen2-7B-Instruct`). The `qwen_1_5` vs `qwen_2` conv templates only affect prompt formatting, not the underlying model. The duplicate weight directory `/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2` was deleted.

---

## Old / Non-OV Backbones

| # | Model | Backbone | Overall | Notes |
|---|-------|----------|:-------:|-------|
| 1 | PruneVID | Legacy | 43.80% | Different backbone/config |
| 2 | VisionZip | LLaVA-v1.5-7b | 39.97% | Wrong backbone |
| 3 | DyTo | LLaVA-v1.6-Vicuna-7b | — | ❌ FAILED v3 (7751031) — LlavaLlamaForCausalLM import from /code/DYTO (mahern69-owned, unpatched). Fix: point PYTHONPATH to /project/rhu/dpalfaro/DYTO (dpalfaro-owned, patched). |

---

## STTM-LLaVAVid Experiment Results (LLaVA-Video-7B-Qwen2 backbone)

| Config | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Notes |
|--------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|-------|
| **t=0.80, 32f** | **53.33%** (2143/4018) | 40.27% (209/519) | 48.83% (188/385) | 57.33% (313/546) | 57.51% (850/1478) | 68.84% (475/690) | 27.00% (108/400) | Job 7706552. Separate backbone. |

---

## Currently Running / Queued (2026-07-23 12:25)

| Job ID | Model | Type | Status |
|:------:|-------|------|--------|
| 7768241 | MDP3 | Fresh re-run (fixed --num_frames) | ⚡ RUNNING |
| 7768829 | VisionZip | Fresh re-run (DyCoke builder fix) | ⚡ RUNNING |
| 7769039 | VisionZip | Fresh re-run (fixed args) | ⚡ RUNNING |

---

## Conv Template Reference

| Template | When to Use |
|----------|-------------|
| `qwen_1_5` | Original template for LLaVA-OV. Standard. |
| `qwen_2` | Alternative prompt format. Produces identical results (minor template-only diff). |

All numbers above use the `qwen_2` template unless otherwise noted. The `qwen_1_5` results are listed separately only for FlashVID where a discrepancy was observed.

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

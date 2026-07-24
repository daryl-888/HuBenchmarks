# Master Results Table — HuBenchmarks

> **Last updated**: 2026-07-23 (late session — smoke-verification pass)
> **Benchmark**: MotionBench — 8,052 MCQ video samples, 4,018 scoreable, 4,034 NA
> **Standard eval**: 32 frames, `do_sample=False`, `max_new_tokens=16`, letter-match scoring, NA-skip

> **Critical Correction (2026-07-23)**: `llava-ov-7b` already uses Qwen2 internally (`LlavaQwenForCausalLM`, `hidden_size=3584`). The "qwen_1_5" vs "qwen_2" distinction is **only a conv template difference** (prompt formatting), not different model weights. The duplicate `llava-ov-7b-qwen2` directory on Carya has been deleted. Both columns in the table below run the **same model** with different templates — identical results are expected and correct.

---

## 📊 Results by Stage

### Stage 1 — LLaVA-OV-7B (`llava-ov-7b`, `LlavaQwenForCausalLM`) *(Mid-2024)*
*12/13 methods complete, VisionZip rerunning*

> **Verification status (2026-07-23)** — every method now smoke-tested with
> `check_run.py --smoke` to confirm it actually engages (not a silent no-op):
>
> | Method | Smoke gate | Note |
> |---|---|---|
> | FastV | ✅ **PASS — genuinely prunes** | rebuilt paper-exact; `img_len=6273 keep=941` (15%); 2/4 preds diverge from baseline |
> | DyCoke | ✅ PASS | fixed: builder defaulted to flash_attn (not installed) → sdpa |
> | HoliTom | ✅ PASS | |
> | AIM | ✅ PASS | |
> | STTM | ✅ PASS | |
> | PruneVID (PLLaVA) | ✅ PASS | real backbone; the *OV port* remains inert |
> | VideoITG | ⚠️ PASS + warn | runs, but infer script emits no `*_params` block |
> | MDP3 | ✅ full run PASS (53.06%) | smoke had `libnccl.so.2` + wrong arg (`--pool-frames`, not `--num_frames`) |
> | FlashVID | 🔧 smoke resubmitted | uses `--frame-counts`, not `--num_frames` |
> | VisionZip | ❌ **FAIL — 100% empty predictions** | same class of bug as FastV's eager-attention failure |
> | DyTo | ❌ **never ran** — 5 consecutive import failures | its one 5.25% "result" emitted free-form captions, not letters; below random. Not a result. |


| # | Model | Conference | Year | Overall | Act. Order | Cam. Motion | Loc. Motion | Mot. Rec. | Mot. Objs. | Rep. Count | Parameters | Notes |
|---|-------|-----------|------|:-------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|------------|-------|
| 1 | **DyCoke** | arXiv | 2024 | **53.36%** (2144/4018) | 38.92% (202/519) | 48.83% (188/385) | 55.68% (304/546) | 58.19% (860/1478) | 70.87% (489/690) | 25.25% (101/400) | l=3, p=0.7, k=0.7 | Best overall. Two-stage token merging + KV cache prune. |
| 2 | **FlashVID** (0.25) | ICLR | 2026 | **53.36%** (2144/4018) | 42.20% (219/519) | 47.53% (183/385) | 55.13% (301/546) | 57.58% (851/1478) | 71.74% (495/690) | 23.75% (95/400) | retention_ratio=0.25 | ICLR 2026 Oral |
| 3 | **FlashVID** (0.15) | ICLR | 2026 | **53.29%** (2141/4018) | 39.69% (206/519) | 45.71% (176/385) | 54.03% (295/546) | 57.85% (855/1478) | 71.88% (496/690) | 28.25% (113/400) | retention_ratio=0.15 | Standard retention ratio |
| 4 | **HoliTom** | — | 2025 | **53.14%** (2135/4018) | 40.85% (212/519) | 49.61% (191/385) | 52.56% (287/546) | 57.04% (843/1478) | 71.45% (493/690) | 27.25% (109/400) | RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5 | |
| 5 | **MDP3** | arXiv | 2025 | **53.06%** (2132/4018) | 40.46% (210/519) | 49.61% (191/385) | 53.48% (292/546) | 56.77% (839/1478) | 71.59% (494/690) | 26.50% (106/400) | pool_frames=32, select_frames=8 | |
| 6 | **VideoITG** | — | 2025 | **52.86%** (2124/4018) | 40.08% (208/519) | 47.01% (181/385) | 53.66% (293/546) | 57.58% (851/1478) | 70.14% (484/690) | 26.75% (107/400) | grounding: 512 sample, 32 select, 2fps | |
| 7 | **AIM** | ICCV | 2025 | **52.84%** (2123/4018) | 41.43% (215/519) | 47.79% (184/385) | 54.03% (295/546) | 57.10% (844/1478) | 71.88% (496/690) | 22.25% (89/400) | aim env, eager attention | Bipartite soft matching + PageRank prune. |
| — | **PruneVID (OV port)** — INERT | — | 2024 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | cluster_ratio=0.5, temporal_segment_ratio=0.25 | ⚠️ Byte-identical to the inert FastV baseline (2116/4018, all categories match). The VTP port to LLaVA-OV did NOT prune — this is the plain backbone, not a PruneVID result. Real PruneVID is on PLLaVA-7B → 44%, see Other Backbones. |
| — | **Backbone baseline** (mislabelled "FastV") | — | 2024 | **52.66%** (2116/4018) | 40.46% (210/519) | 45.19% (174/385) | 55.49% (303/546) | 57.04% (843/1478) | 71.16% (491/690) | 23.75% (95/400) | k=2, r=0.5 (INERT) | ⚠️ NOT a FastV result. `apply_fastv()` is a stub — `enabled: false` in summary.json. This is the plain LLaVA-OV-7B backbone. Do not rank as a method. |
| 10 | **STTM-v2** | — | 2025 | **51.87%** (2084/4018) | 39.11% (203/519) | 48.83% (188/385) | 53.66% (293/546) | 53.59% (792/1478) | 71.16% (491/690) | 29.25% (117/400) | sa_start_layer_idx=2, sa_tree_thresh=0.85 | |
| 11 | **FlashVID** (0.15, qwen15) | ICLR | 2026 | **51.22%** (2058/4018) | 39.50% (205/519) | 41.04% (158/385) | 54.95% (300/546) | 54.19% (801/1478) | 71.16% (491/690) | 25.75% (103/400) | retention_ratio=0.15, qwen_1_5 template | qwen_1_5 template run only. |
| 12 | **VideoITG** (simplified) | — | 2025 | **34.89%** (1402/4018) | 14.45% (75/519) | 45.19% (174/385) | 55.49% (303/546) | 44.65% (660/1478) | 20.29% (140/690) | 12.50% (50/400) | single-stage, no grounding | |
| 13 | **VisionZip** | — | 2024 | — | — | — | — | — | — | — | dominant=54, contextual=10 | 🔄 Rerunning (7768829, 7769201) |

### Stage 2 — LLaVA-Video-7B (`llava-video-7b`) *(Late-2024)*
*ALL CANCELLED 2026-07-23 — user descoped Stage 2. Jobs 7769192-7769209 killed to free nodes.*

| Method | Overall | Status | Job ID |
|--------|:-------:|:------:|:------:|
| AIM | — | ⚡ RUNNING | 7769192 |
| FastV | — | ⚡ RUNNING | 7769194 |
| DyCoke | — | ⏳ PENDING | 7769193 |
| FlashVID | — | ⏳ PENDING | 7769195 |
| HoliTom | — | ⏳ PENDING | 7769196 |
| MDP3 | — | ⏳ PENDING | 7769197 |
| VideoITG | — | ⏳ PENDING | 7769198 |
| STTM | — | ⏳ PENDING | 7769209 |

*Not applicable to Stage 2: FastVID (model bug), PruneVID (PLLaVA-7B), VisionZip (LLaVA-1.5), DyTo (Vicuna)*

### Stage 3 — Qwen3-VL-8B (`qwen3-vl-8b`) *(Late-2025)*

> **UNBLOCKED 2026-07-23.** Stage 3 was never runnable: every pre-existing conda env
> has transformers 4.45, which lacks `Qwen3VLForConditionalGeneration`. New env
> **`qwen3vl`** (torch 2.6.0+cu124, transformers 5.14.1) loads it. Baseline verified
> (smoke 7770277 = 50% on 8 samples) and the **full baseline is running (7770357)**.
>
> **Frame bug found+fixed:** `Qwen3VLVideoProcessor` has `do_sample_frames=True, fps=2`,
> so it RE-SAMPLED our frames and ignored `--num_frames` (warning: "Defaulting to
> fps=24"). Fixed with `do_sample_frames=False`; verified `requested=32 given=32`.
>
> **All 8 existing "ports" are fake** — each `eval_<m>.py` is the same baseline script
> applying no method. They need real re-implementation, not submission.

| Method | Overall | Status |
|--------|:-------:|:------:|
| AIM | — | 📋 Ported, not submitted |
| DyCoke | — | 📋 Ported, not submitted |
| FastV | — | 📋 Ported, not submitted |
| FlashVID | — | 📋 Ported, not submitted |
| HoliTom | — | 📋 Ported, not submitted |
| MDP3 | — | 📋 Ported, not submitted |
| STTM | — | 📋 Ported, not submitted |
| VideoITG | — | 📋 Ported, not submitted |

### Other Backbones
*Methods with non-LLaVA backbones (separate evaluation track)*

| Method | Backbone | Overall | Status | Job ID |
|--------|----------|:-------:|:------:|:------:|
| DyTo | LLaVA-NeXT Vicuna-7B | — | 🔄 PENDING (builder patched) | 7769212 |
| PruneVID | PLLaVA-7B | **44.00%** (1768/4018) | ✅ Real result, `pruning_enabled: True` (prunevid_run2) | — |
| VisionZip | LLaVA-v1.5-7b | 39.97% | 🔄 Stage 1 rerun (Vicuna LLM) | 7768829, 7769201 |
| STTM-LLaVAVid | LLaVA-Video-7B | **53.33%** | ✅ Experiment run (job 7706552) | — |
| iMove | LLaVA-NeXT | — | ❌ No public code | — |
| TrajViT | LLaVA-NeXT | — | ❌ No public code | — |

---

## Template Comparison: qwen_1_5 vs qwen_2

| Model | qwen_1_5 | qwen_2 | Delta | Notes |
|-------|:--------:|:------:|:-----:|-------|
| DyCoke | **53.36%** | **53.36%** | 0.00 | Identical — template has no effect |
| HoliTom | **53.14%** | **53.14%** | 0.00 | Identical |
| MDP3 | **53.06%** | **53.06%** | 0.00 | Identical |
| Backbone baseline (via FastV stub) | **52.66%** | **52.66%** | 0.00 | Identical — plain backbone, both templates |
| FlashVID (0.15) | **51.22%** | **53.29%** | -2.07 | ⚠️ Different results! Only method with diff |

FlashVID's result difference (51.22% vs 53.29%) suggests one of its runs may have had different hyperparameters or a different code path — not just a template difference. Worth investigating.

---

## Currently Running / Queued (2026-07-23 13:58 CDT)

| Job ID | Model | Stage | Status | Time |
|:------:|-------|:-----:|:------:|:----:|
| 7768241 | MDP3 OV1.5 rerun | Stage 1 | ⚡ RUNNING | ~3h |
| 7768829 | VisionZip v1 | Stage 1 | ⚡ RUNNING | ~2.5h |
| 7769192 | stage2_aim | Stage 2 | ⚡ RUNNING | ~20m |
| 7769194 | stage2_fastv | Stage 2 | ⚡ RUNNING | ~20m |
| 7769212 | DyTo (vicuna) | Other | ⏳ PENDING | — |
| 7769193 | stage2_dycoke | Stage 2 | ⏳ PENDING | — |
| 7769195 | stage2_flashvid | Stage 2 | ⏳ PENDING | — |
| 7769196 | stage2_holitom | Stage 2 | ⏳ PENDING | — |
| 7769197 | stage2_mdp3 | Stage 2 | ⏳ PENDING | — |
| 7769198 | stage2_videoitg | Stage 2 | ⏳ PENDING | — |
| 7769201 | VisionZip v3 | Stage 1 | ⏳ PENDING | — |
| 7769209 | stage2_sttm | Stage 2 | ⏳ PENDING | — |

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

# Master Results — HuBenchmarks / MotionBench

**Updated** 2026-07-24 · **Benchmark** MotionBench (8,052 samples · 4,018 scoreable · 4,034 NA)
**Standard config** 32 frames · retention 0.15 where the method exposes one · `do_sample=False` · `max_new_tokens=16` · letter-match scoring · NA skipped

**Goal** — 11 methods × 2 backbones (LLaVA-OV-7B, Qwen3-VL-8B) = **22 cells**, each a paper-exact implementation verified to actually engage.

> ## 🏆 Headline
> **Qwen3-VL-8B baseline = 62.52%** vs **LLaVA-OV-7B baseline = 52.66%** — the newer
> backbone is ~10 points stronger on MotionBench. This is the single most important
> result: on this benchmark the backbone dominates the efficiency method.

---

## 1. Legend

| Mark | Meaning |
|:---:|---|
| ✅ | Verified: method provably engages (smoke-gated, predictions diverge from baseline) |
| 🟡 | Provisional: number exists but predates the verification harness — **not** gated |
| ❌ | Invalid: silent no-op, crash, or below random. **Not a result.** |
| 🔄 | In flight |
| 📋 | Not yet implemented |

> **Read this before quoting any number.** Section 3 now holds **Wave 2 gated numbers**
> (2026-07-24): each was re-run with the `enabled` flag AND confirmed to diverge from the
> plain backbone (differ counts below). These are the trustworthy figures. Numbers marked
> ❌ are known-invalid and must not be reported.

---

## 2. Verification status — LLaVA-OV-7B (10/11 methods verified)

| Method | Engages? | Evidence / fix required to get there |
|---|:---:|---|
| **FastV** | ✅ | Rebuilt paper-exact from the authors' code. `img_len=6273 keep=941` (15%); 2/4 predictions diverge from baseline |
| **DyCoke** | ✅ | Builder defaulted to flash_attn (absent) → sdpa |
| **HoliTom** | ✅ | — |
| **AIM** | ✅ | — |
| **STTM** | ✅ | — |
| **MDP3** | ✅ | `libnccl.so.2` lives in the conda env, not `mdp3_pkgs`; arg is `--pool-frames` |
| **PruneVID** | ✅ | On its real PLLaVA backbone. The *LLaVA-OV port* is inert (see ❌ below) |
| **VisionZip** | ✅ | `output_ids[:, input_ids.shape[1]:]` discarded the whole response → 100% empty. Slice removed |
| **FlashVID** | ✅ | 3 bugs: `--frame-counts` not `--num_frames`; motionbench variant loaded `LlavaLlamaForCausalLM` at retention 0.10; flash_attn → sdpa |
| **VideoITG** | ✅ | Two-stage grounding; now emits `videoitg_params`. Wave 2 gated at 52.86% |
| **DyTo** | 🔄 | Import FIXED (was `try/except: pass` swallowing an absolute-import error → aliased `dyto.llava` as `llava`). Then hit a 2nd bug: conv template `image_seq_v3` doesn't exist → `vicuna_v1`. Re-verifying (7776371) |

---

## 3. Stage 1 — LLaVA-OV-7B ✅ Wave 2 GATED (2026-07-24)

Each row re-run with the standard config and confirmed to **diverge from the plain
backbone** (differ = #predictions ≠ the inert baseline, out of 8,052). All 8,052 samples;
4,018 scoreable. Backbone baseline = **52.66%**.

Subcategory columns are % correct within that category. Category totals (scoreable):
Action Order 519 · Camera Motion 385 · Location-related Motion 546 · Motion Recognition
1478 · Motion-related Objects 690 · Repetition Count 400.


> ### ⚠️ These rankings are NOT statistically significant
> McNemar's test on paired predictions, DyCoke (best) vs baseline:
> `baseline-wrong→dycoke-right = 170`, `baseline-right→dycoke-wrong = 142`,
> **χ² = 2.34** (needs ≥3.84 for p<0.05). The +0.70 gain is **noise**. The same
> holds for FlashVID (+0.65), HoliTom (+0.48), MDP3 (+0.40).
> **Read this as characterization, not a leaderboard.** Six of seven methods are accuracy-neutral within the ±1.54 pt resolution floor. The supportable claim is
> that on LLaVA-OV these methods are indistinguishable from the backbone and from
> each other. See [DETERMINISM_AND_VALIDITY.md](../../docs/DETERMINISM_AND_VALIDITY.md).

### 3a. Results — accuracy, venue, subcategories

| # | Method | Venue | Year | Overall | Differ | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|:-:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **DyCoke** ✅ | arXiv 2411.14401 | 2024 | **53.36%** (2144) | 1031 | 38.9 | 48.8 | 55.7 | 58.2 | 70.9 | 25.2 |
| 2 | **FlashVID** ✅ | ICLR (Oral) | 2026 | **53.31%** (2142) | 1610 | 39.9 | 45.7 | 53.8 | 58.0 | 71.7 | 28.2 |
| 3 | **HoliTom** ✅ | — | 2025 | **53.14%** (2135) | 1838 | 40.8 | 49.6 | 52.6 | 57.0 | 71.4 | 27.2 |
| 4 | **MDP3** ✅ | ICCV | 2025 | **53.06%** (2132) | 1452 | 40.5 | 49.6 | 53.5 | 56.8 | 71.6 | 26.5 |
| 5 | **AIM** ✅ | ICCV | 2025 | **52.86%** (2124) | 1406 | 41.4 | 48.1 | 54.2 | 57.0 | 71.9 | 22.2 |
| 5 | **VideoITG** ✅ | — | 2025 | **52.86%** (2124) | 1193 | 40.1 | 47.0 | 53.7 | 57.6 | 70.1 | 26.8 |
| — | *Backbone baseline* | — | — | *52.66%* (2116) | *0 (ref)* | *40.5* | *45.2* | *55.5* | *57.0* | *71.2* | *23.8* |
| 7 | **STTM** ✅ | — | 2025 | **51.72%** (2078) | 4859 | 39.9 | 48.1 | 53.8 | 53.5 | 70.6 | 28.8 |
| 8 | **VisionZip** ✅ | — | 2024 | **40.09%** (1611) | 4506 | 33.7 | 33.0 | 37.2 | 41.1 | 57.4 | 25.8 |
| 9 | **FastV** ✅ | arXiv 2403.06764 | 2024 | **36.78%** (1478) | 4605 | 32.8 | 31.9 | 34.1 | 35.7 | 53.8 | 25.2 |

*Numbering is presentation order only — see the significance warning above.*
*Differ = predictions ≠ the plain backbone, out of 8,052 (0 would mean the method never ran).*

### 3b. Per-model detail — mechanism, parameters, notes

| Method | Mechanism | Parameters used | Notes |
|---|---|---|---|
| **DyCoke** | Two-stage: temporal token merging across frames, then dynamic KV-cache pruning at LLM layer *l* | `l=3, p=0.7, k=0.7` | Highest score, but +0.70 over baseline is **not significant** (χ²=2.34). Builder defaulted to flash-attn (absent) → forced `sdpa` |
| **FlashVID** | Pre-LLM temporal-segment merge; score = α·saliency + (1−α)·distinctiveness | `retention_ratio=0.15, alpha=0.7, T=0.8` | Three bugs fixed: retention was **hardcoded 0.25** ignoring the CLI; the alt script loaded `LlavaLlamaForCausalLM` (wrong class); arg is `--frame-counts` not `--num_frames` |
| **HoliTom** | Outer (pre-LLM) temporal-segment retain + inner (in-LLM) attention merge from layer *k* | `RETAIN_RATIO=0.15, T=0.80, k=18, r=0.5` | Env-var driven. Requires `transformers==4.45.2` exactly |
| **MDP3** | Conditional determinantal point process frame selection, **before** the model | `pool_frames=32, select_frames=8` | Only genuinely model-agnostic method here. `libnccl.so.2` lives in the conda env, not `mdp3_pkgs`; arg is `--pool-frames` |
| **AIM** | Bipartite soft-matching merge (4 steps: 50→25→12.5→6.25%) + PageRank pruning | compiled into patched `llava_arch.py` (no CLI knobs) | Params not CLI-configurable; recorded in the summary for gate-ability |
| **VideoITG** | Two-stage grounded frame selection: grounding pass scores frames, inference pass consumes chosen indices | grounding: 512 sampled, 32 selected, 2 fps | Needs `frame_scores.jsonl` from stage 1. Smoke initially called the wrong script (`eval_videoitg.py` vs `..._infer.py`) |
| **STTM** | Quadtree spatio-temporal merging patched into Qwen2 attention | `sa_start_layer_idx=2, sa_tree_thresh=0.85` | Efficiency comes from LLM layers; vision tower runs normally |
| **VisionZip** | Dominant tokens (CLS-attention) + contextual tokens (key-vector similarity merge) | `dominant=54, contextual=10` | **Was 0% across 3 runs** — `output_ids[:, input_ids.shape[1]:]` discarded every response (LLaVA-1.5 returns only new tokens). Fixed → 40.09% |
| **FastV** | Attention-rerank: rank image tokens at layer K by received attention, keep top fraction | `k=2, r=0.85` (keeps 15%) | Was a **no-op stub** (`enabled:false`) reporting the bare backbone. Rebuilt paper-exact. Paper default is `r=0.5`; **our 15% is far more aggressive**, which explains the −15.9 drop |
| **PruneVID** | Video Token Pruning: DPC-KNN clustering → temporal segments → cluster-centroid merge | `cluster_ratio=0.5, temporal_segment_ratio=0.25, layer=10` | Runs on **PLLaVA-7B** (its published backbone) = 44.13%. The LLaVA-OV port was inert until switched to `PrunableDynamicCache.kv_cache`; full run in flight |
| **DyTo** | FINCH clustering (~25 of 100 frames) + ToMe dynamic merge | `temporal_aggregation=spatial_tome_finch_dynamic_all_frms, rope_scaling=2` | ❌ **Not reproducible from published artifacts** — two defects in released code ([evidence](../../docs/UPSTREAM_DEFECTS.md)) |

### ❌ Invalid entries — do not report

| Label | Number | Why it is not a result |
|---|:---:|---|
| "FastV" (old) | 52.66% | `apply_fastv()` was a stub, `enabled: false`. Bare backbone |
| "PruneVID (OV port)" | 52.66% | **0/8052** differ from the inert run. VTP never fired |
| "VisionZip" (old) | 0.00% ×3 | Output-slicing bug (fixed) discarded every response |
| "VideoITG (simplified)" | 34.89% | Single-stage, no grounding — not the method |
| "FlashVID (qwen15)" | 51.22% | Ran at retention 0.10 with the wrong model class |

**PruneVID** (real, on PLLaVA-7B) = **44.13%** — see §5 Other backbones.

---

## 4. Stage 3 — Qwen3-VL-8B

> **Baseline verified: 62.52%** (2512/4018) — full run, 32 frames confirmed. This is the
> `--vs-baseline` reference for every port, and ~10 points above the LLaVA-OV baseline.
>
> Path to here: (1) new env **`qwen3vl`** (torch 2.6.0+cu124, transformers 5.14.1) — every
> older env lacked `Qwen3VLForConditionalGeneration`; (2) frame fix — the processor's
> `do_sample_frames=True, fps=2` was re-sampling and ignoring `--num_frames`; set
> `do_sample_frames=False`, verified `requested=32 given=32`.

### Baseline subcategories — Qwen3-VL vs LLaVA-OV

| Backbone | Overall | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-VL-8B** | **62.52%** (2512) | 46.1 | 63.1 | 65.0 | 67.3 | 79.0 | 33.8 |
| LLaVA-OV-7B | 52.66% (2116) | 40.5 | 45.2 | 55.5 | 57.0 | 71.2 | 23.8 |
| **Δ (Qwen3-VL − OV)** | **+9.86** | +5.6 | **+17.9** | +9.5 | +10.3 | +7.8 | +10.0 |

Qwen3-VL beats LLaVA-OV in **every** subcategory — largest gap on **Camera Motion (+17.9)**,
smallest on Action Order (+5.6). Even the hardest category for both (Repetition Count) rises
from 23.8% → 33.8%.

### Method ports

| Method | Venue | Year | Overall | Smoke gate | Parameters | Notes |
|---|:---:|:---:|:---:|:---:|---|---|
| **Baseline** | — | — | **62.52%** (2512) | ✅ | 32 frames, no method | Reference for every divergence check |
| FastV | arXiv | 2024 | 🔄 full run | ✅ | k=2, r=0.85 | `ACTIVE: visual=11664 keep=1750`. Attention recomputed from q/k (sdpa returns none) |
| DyCoke | arXiv | 2024 | 🔄 full run | ✅ | l=3, p=0.7, k=0.7 | `stage1=8165 stage2=5716 (net 49%)` |
| HoliTom | — | 2025 | 🔄 full run | ✅ | RETAIN=0.15, T=0.80, k=18, r=0.5 | `outer=1750 inner=875 (net 7.5%)` |
| FlashVID | ICLR (Oral) | 2026 | 🔄 full run | ✅ | retention=0.15, α=0.7 | `keep=1750 (15.0%)`. Embedding-only → no eager attention needed |
| AIM | ICCV | 2025 | 🔄 full run | ✅ | 4 merge steps + PageRank | `after_merge=1750 keep=1750` |
| MDP3 | ICCV | 2025 | 🔄 full run | ✅ | pool=32, select=8 | Selector loaded by file path — `vlmeval` imports a symbol removed in transformers 5.x |
| VideoITG | — | 2025 | **56.35%** (2264) ✅ | ✅ | 512 sampled, 32 selected | **−6.17 vs baseline, χ²=92.4 → SIGNIFICANT loss.** Subcats 45.3/53.5/59.3/60.2/75.4/22.2. Stage-1 grounding indices reused verbatim |
| VisionZip (contextual-only) | 🔄 full run pending | ✅ | 5/8 divergence, 11664->1750 (15%). Hook on `vis.merger`; token count preserved for masked_scatter. Partial by design (no CLS) |
| PruneVID | 🟡 blocked | — | VTP core IS separable, but the LLaVA-OV port is still inert — fix that first |
| STTM | ❌ | — | ships a wholesale Qwen2Model_forward replacement; no separable merge fn |
| DyTo | ❌ | — | mechanism lives in LLaVA's llava_arch.py; Vicuna-bound |

**7 full runs in flight** (~10h each). Getting them to engage required untangling a
5-bug chain; two ports printed `ACTIVE` while producing byte-identical output and
only the divergence gate caught it. See PORT_FEASIBILITY.md and METHODOLOGY.md.

---

## 5. Other backbones (separate track)

| Method | Backbone | Overall | Status |
|---|---|:---:|---|
| PruneVID | PLLaVA-7B | **44.13%** ✅ | Wave 2 gated (`enabled: True`). **Not run on LLaVA-OV**: its VTP is bound to PLLaVA — the OV port produced 0/8052 divergence (inert). PLLaVA-7B is its published backbone. |
| VisionZip | LLaVA-1.5-7B | **39.97%** 🟡 | Only usable VisionZip figure; the 0.00% runs are ❌ |
| STTM-LLaVAVid | LLaVA-Video-7B | **53.33%** 🟡 | `sttm_llavavid_t80_full` |
| DyTo | LLaVA-NeXT Vicuna-7B | ❌ 5.25% | Emitted captions, not letters. Below random. Not a result |
| iMove, TrajViT | — | — | ❌ No public code / weights |

*Stage 2 (LLaVA-Video-7B) was descoped 2026-07-23; those jobs were cancelled.*

---

## 6. Known duplicate rows (to dedupe)

Verified at prediction level — **0/8052 differ**, i.e. the same run recorded under
several names (the old "ovqwen15 vs ovqwen2" split was the *same model*):

* DyCoke 53.36% → `ovqwen_dycoke_run1`, `dycoke_ovqwen15_fresh`, `dycoke_ovqwen2_fresh`
* MDP3 53.06% → `mdp3_qwen15_run`, `mdp3_ovqwen15_fresh`, `ovqwen2_mdp3_run1`
* HoliTom 53.14%, VideoITG 52.86% → 2 dirs each

⚠️ `flashvid_run4` ties DyCoke's score exactly but differs in **1240/8052** predictions —
same number, different behaviour. **Score-matching alone is never evidence.**

---

## 7. Reference

**Subcategories** — Action Order 519 · Camera Motion 385 · Location-related Motion 546 ·
Motion Recognition 1,478 · Motion-related Objects 690 · Repetition Count 400 · **Total 4,018**
(+4,034 NA = 8,052 samples).

**Conv template** — `qwen_1_5` and `qwen_2` differ only in prompt formatting, not weights.
`llava-ov-7b` already uses Qwen2 internally (`LlavaQwenForCausalLM`); the duplicate
`llava-ov-7b-qwen2` weights directory was a byte-for-byte copy and has been deleted.

**Verification** — `python3 scripts/check_run.py <run> --expect-method <m> [--smoke] [--vs-baseline <ref>]`.
0 differing predictions vs the backbone ⇒ silent no-op ⇒ the method did not run.

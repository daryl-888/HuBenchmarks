# Master Results — HuBenchmarks / MotionBench

**Updated** 2026-07-25 (**all 10 portable Qwen3-VL methods gated**; Stage 3 complete) · **Benchmark** MotionBench (8,052 samples · 4,018 scoreable · 4,034 NA)
**Standard config** 32 frames · retention 0.15 where the method exposes one · `do_sample=False` · `max_new_tokens=16` · letter-match scoring · NA skipped

**Goal** — 11 methods × 2 backbones (LLaVA-OV-7B, Qwen3-VL-8B) = **22 cells**, each a paper-exact implementation verified to actually engage.
**Status** — 10/11 gated on LLaVA-OV · 10/11 gated on Qwen3-VL, **all 10 portable methods gated, PruneVID + STTM included** (2026-07-25) · DyTo runs on its own Vicuna backbone as a labelled variant. **All 11 methods now have a working implementation on both backbones**; DyTo's full run is in flight (job 7833211, submitted 2026-07-25).

> ## 🏆 Headline findings
>
> **1. The backbone dominates the method.** Qwen3-VL-8B baseline **62.52%** vs
> LLaVA-OV-7B **52.66%** — a **+9.86** gap (χ²=127.5, highly significant). No
> efficiency method on either backbone comes close to that.
>
> **2. Method effects do NOT transfer across backbones.** On LLaVA-OV **no**
> method's change is significant — the methods are effectively free. On Qwen3-VL
> **all 10 lose accuracy and 8 lose significantly**, at the same 32 frames and 15%
> retention. VideoITG is the sharpest case: +0.20 (ns) on LLaVA-OV, **−6.17**
> (χ²=92.4) on Qwen3-VL. Reduction that is harmless on a weaker backbone actively
> harms a stronger one — a ranking measured on one backbone must not be assumed on
> another.
>
> **3. On LLaVA-OV, no method significantly beats the backbone.** (The other half of finding 2.) All nine sit
> within the ±1.54 pt resolution floor; only FastV and VisionZip differ
> significantly, and both are *worse* at the standardized 15% budget.

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

## 2. Verification status — LLaVA-OV-7B (10/11 verified; DyTo runs as a labelled variant)

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
| **DyTo** | 🟡 | **Runs as a labelled variant** (2026-07-25). Four of our bugs fixed, then two *upstream* defects: `finch_cluster()`'s missing `return` was **recovered by verbatim transcription** from the sibling KMeans function (13/14 normalized lines identical); `FINCH(tw_finch=...)` needed a **reconstruction** of the published TW-FINCH rule via `initial_rank`. Smoke: 0 tracebacks, real letters. Report only as **"DyTo (reconstructed TW-FINCH)"** ([evidence](../../docs/UPSTREAM_DEFECTS.md)) |

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
| 10 | **PruneVID** (OV port) ✅ | — | 2024 | **38.20%** (1535) | 4536 | 32.0 | 34.8 | 35.5 | 38.4 | 52.9 | 27.2 |

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
| **PruneVID** | Video Token Pruning: DPC-KNN clustering → temporal segments → cluster-centroid merge | `cluster_ratio=0.5, temporal_segment_ratio=0.25, layer=10` | On **PLLaVA-7B** (published backbone) = 44.13%. **LLaVA-OV port = 38.20%** — was an inert no-op until switched to `PrunableDynamicCache.kv_cache`; now engages (4536/8052 differ) and loses **−14.46 vs backbone, χ²=220.3 (significant)** at 50% retention. Trips the gate's 0.40 accuracy floor, but that is a **false alarm**: all 4 letters used, only 8 empty preds — a real degradation, not a broken run |
| **DyTo** | FINCH clustering (~25 of 100 frames) + ToMe dynamic merge | `temporal_aggregation=spatial_tome_finch_dynamic_all_frms, rope_scaling=2` | 🟡 Runs as **"DyTo (reconstructed TW-FINCH)"**. **A/B measured:** TW-FINCH vs standard FINCH = **0/8 differ** (jobs 7786505 vs 7786513) — the clustering choice is not observable in the output at n=8; DyTo's ToMe merge and 25-frame cap absorb it. Do not claim the weighting changes results without a larger paired test ([evidence](../../docs/UPSTREAM_DEFECTS.md)) |

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

### 4a. Results — accuracy, venue, subcategories ✅ ALL 10 GATED (2026-07-25)

All 8,052 samples · 32 frames verified · 8 empty predictions each (known bad-NFS
videos) · 0 tracebacks · ACTIVE log **and** nonzero divergence on every row.
Backbone baseline = **62.52%**.

| # | Method | Venue | Year | Overall | Differ | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|:-:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| — | *Backbone baseline* | — | — | *62.52%* (2512) | *0 (ref)* | *46.1* | *63.1* | *65.0* | *67.3* | *79.0* | *33.8* |
| 1 | **PruneVID** ✅ | — | 2024 | **62.17%** (2498) | 1074 | 46.1 | 62.9 | 64.1 | 67.1 | 79.1 | 32.5 |
| 2 | **DyCoke** ✅ | arXiv 2411.14401 | 2024 | **61.85%** (2485) | 1048 | 45.1 | 62.6 | 63.9 | 66.4 | 78.4 | 34.5 |
| 3 | **HoliTom** ✅ | — | 2025 | **60.33%** (2424) | 1657 | 44.7 | 61.3 | 61.0 | 66.4 | 76.2 | 29.0 |
| 4 | **MDP3** ✅ | ICCV | 2025 | **59.66%** (2397) | 1626 | 44.9 | 64.7 | 62.8 | 64.0 | 77.4 | 23.0 |
| 5 | **FastV** ✅ | arXiv 2403.06764 | 2024 | **59.01%** (2371) | 2029 | 46.2 | 61.0 | 61.9 | 63.5 | 72.2 | 30.5 |
| 6 | **VisionZip** (contextual-only) ✅ | — | 2024 | **58.81%** (2363) | 2035 | 43.2 | 57.4 | 61.9 | 64.2 | 74.3 | 29.5 |
| 7 | **STTM** ✅ | — | 2025 | **57.07%** (2293) | 2095 | 44.3 | 60.0 | 61.0 | 60.6 | 73.8 | 23.8 |
| 8 | **FlashVID** ✅ | ICLR (Oral) | 2026 | **56.65%** (2276) | 2625 | 43.4 | 57.9 | 57.1 | 61.2 | 73.0 | 27.0 |
| 9 | **VideoITG** ✅ | — | 2025 | **56.35%** (2264) | 2522 | 45.3 | 53.5 | 59.3 | 60.2 | 75.4 | 22.2 |
| 10 | **AIM** ✅ | ICCV | 2025 | **55.97%** (2249) | 2861 | 45.9 | 56.6 | 59.5 | 58.3 | 72.2 | 27.0 |

*Differ = predictions ≠ the plain backbone, out of 8,052 (0 would mean the method never ran).*
*VisionZip is the **contextual-only partial** — Qwen3-VL has no CLS token, so the dominant
half is not implementable. Never quote row 5 as plain "VisionZip"; the complete method is
the LLaVA-OV row (40.09%).*

**Significance vs baseline** (McNemar on paired predictions; χ² ≥ 3.84 ⇒ p<0.05):

| Method | Δ vs base | Win / Lose | χ² | Significant? |
|---|:---:|:---:|:---:|:---:|
| **PruneVID** | −0.35 | 56 / 70 | 1.3 | **no — smallest loss of all 10** |
| **DyCoke** | −0.67 | 83 / 110 | 3.5 | **no** |
| **HoliTom** | −2.19 | 131 / 219 | 21.6 | **yes** |
| **MDP3** | −2.86 | 159 / 274 | 30.0 | **yes** |
| **FastV** | −3.51 | 149 / 290 | 44.6 | **yes** |
| **VisionZip** | −3.71 | 191 / 340 | 41.3 | **yes** |
| **STTM** | −5.45 | 170 / 389 | 85.0 | **yes** |
| **FlashVID** | −5.87 | 178 / 414 | 93.3 | **yes** |
| **VideoITG** | −6.17 | 206 / 454 | 92.4 | **yes** |
| **AIM** | −6.55 | 194 / 457 | 105.4 | **yes** |

> ### ⚠️ Every method loses, and loss tracks aggressiveness
> **8 of 10 lose significantly.** Only **PruneVID (−0.35, χ²=1.3)** and
> **DyCoke (−0.67, χ²=3.5)** are statistically indistinguishable from the backbone —
> and both are the most conservative, retaining **50%** and **49%** of tokens
> respectively where the others cut to 15%. AIM and FlashVID prune hardest and lose most
> (−6.55, −5.87). Repetition Count is the weakest category everywhere (23–34%).
>
> **This inverts the Stage-1 picture.** On LLaVA-OV *no* method's change was
> significant — the methods were effectively free. On Qwen3-VL, the same methods at
> the same 32 frames and 15% retention impose real, measurable costs. Frame and
> token reduction that is harmless on a weaker backbone actively harms a stronger
> one: Qwen3-VL extracts more from the full token set, so discarding it costs more.
>
> **Method rankings do not transfer across backbones.** This now rests on 8 gated
> cells rather than the single VideoITG data point that first suggested it.

### 4b. Method ports — parameters, engagement, notes

| Method | Parameters | ACTIVE signature (proves engagement) | Notes |
|---|---|---|---|
| **VideoITG** | 512 sampled, 32 selected, 2 fps | grounding loaded for 8052 samples | Stage-1 grounding indices reused verbatim from the LLaVA-OV run — they are model-agnostic |
| **FastV** | `k=2, r=0.85` (keep 15%) | `visual=11664 keep=1750 (dropped 9914)` | Attention **recomputed from q/k**: sdpa returns `attn_weights=None` and loading eager breaks generation |
| **DyCoke** | `l=3, p=0.7, k=0.7` | `stage1=8165 stage2=5716 (net 49.0%)` | Two-stage; net retention is higher than the 15% standard because `p`·`k` is the paper's own setting. **The only non-significant loss** — consistent with it pruning least |
| **HoliTom** | `RETAIN=0.15, T=0.80, k=18, r=0.5` | `outer=1750 inner=875 (net 7.5%)` | Most aggressive net reduction of the set, yet loses less than AIM/FlashVID |
| **FlashVID** | `retention=0.15, α=0.7, T=0.8` | `keep=1750 (15.0%) segments=11156` | Embedding-only → runs under sdpa, no attention needed |
| **AIM** | 4 merge steps + PageRank | `after_merge=1750 keep=1750 steps=4` | Embedding-only → sdpa. **Largest loss of the set** (−6.55) |
| **MDP3** | `pool=32, select=8` | `pool=32 -> selected=8 frames` | Selector loaded **by file path** — `vlmeval` imports `AutoModelForVision2Seq`, removed in transformers 5.x which Qwen3-VL requires |
| **VisionZip** (contextual-only) | `contextual=1750` (15%) | `merger out 11664 -> 1750 (15.0%)` | **Partial by design.** Dominant half needs a CLS token Qwen3-VL lacks. Hooked on `vis.merger`; token count preserved for `masked_scatter`. Report only as "VisionZip (contextual-only)" |

### 4c. Ports completed 2026-07-25 — both now GATED on the full 8,052

Both passed the full gate with **zero warnings**: 8052/8052 samples, NA accounting
sane, accuracy in band, params present, non-degenerate predictions, and nonzero
divergence. PruneVID **62.17%** (1074 differ), STTM **57.07%** (2095 differ).

| Method | Smoke gate | ACTIVE signature | Notes |
|---|:---:|---|---|
| **PruneVID** | ✅ 3/8 differ | `visual=11664 kept=5832 (50.0%) segments=9` | Authors' own `cluster_dpc_knn`; `visual_pos_masks` for exact positions (no `start=14` offset); additive-mask path, since LLaVA-OV's `kv_cache` mechanism does not exist here |
| **STTM** | ✅ 3/8 differ | `visual=11664 merged=1050 (9.0%) grid=16x27x27 runs=16 seq 11853→1239 deepstack=3` | Authors' own `get_quadtree_features`. **The only port that SHORTENS the sequence** rather than masking, so `position_ids` / `visual_pos_masks` / `deepstack_visual_embeds` are all rebuilt |

> #### ⚠️ STTM's first attempt was a silent no-op — the gate caught it
> It printed nothing and produced **0/8 divergence**, because I assumed the visual
> span was contiguous. Diagnostic measurement showed otherwise: **11,664 visual
> tokens spread over an 11,784-wide span with 15 gaps at a perfectly regular 737
> stride** — Qwen3-VL interleaves **timestamp text tokens between frames**, giving
> 16 planes of 729 tokens separated by 8-token blocks. The rebuild now splices per
> contiguous run. This is the third time a port printed plausible output while
> doing nothing, and the third time only the divergence check caught it.

Getting these ports to engage required untangling a 5-bug chain; two printed
`ACTIVE` while producing byte-identical output, and only the divergence gate caught
it — which is why every row above reports a Differ count. See [PORT_FEASIBILITY.md](../../stage3-qwen3-vl/PORT_FEASIBILITY.md) and
[METHODOLOGY.md](../../docs/METHODOLOGY.md).

---

## 5. Other backbones (separate track)

| Method | Backbone | Overall | Status |
|---|---|:---:|---|
| PruneVID | PLLaVA-7B | **44.13%** ✅ | Wave 2 gated (`enabled: True`). **Not run on LLaVA-OV**: its VTP is bound to PLLaVA — the OV port produced 0/8052 divergence (inert). PLLaVA-7B is its published backbone. |
| VisionZip | LLaVA-1.5-7B | **39.97%** 🟡 | Only usable VisionZip figure; the 0.00% runs are ❌ |
| STTM-LLaVAVid | LLaVA-Video-7B | **53.33%** 🟡 | `sttm_llavavid_t80_full` |
| DyTo (reconstructed TW-FINCH) | LLaVA-NeXT Vicuna-7B | 🔄 full run in flight (7833211) | Smoke clean: 0 tracebacks, real letters `B C A C D B B A`, 0 empty. **Never report as plain "DyTo"** — see §5a. The earlier ❌ 5.25% run (captions, not letters) is superseded and must not be cited |
| iMove, TrajViT | — | — | ❌ No public code / weights |

*Stage 2 (LLaVA-Video-7B) was descoped 2026-07-23; those jobs were cancelled.*

---

### 5a. DyTo — what "reconstructed TW-FINCH" means, and why it is not "DyTo"

DyTo's published artifacts contain two defects. They are **not** equally
recoverable, and the distinction decides how the number may be reported.

| Defect | Resolution | Status |
|---|---|---|
| `finch_cluster()` has **zero return statements** — returns `None`, caller dereferences `.shape` | The same file holds a sibling KMeans function, **character-identical over the whole shared region** (13/14 normalized lines; only difference a stray space before a colon). Its 8-line tail is missing here, `new_embeddings` is declared and never used, and the caller needs a stacked 3-D tensor. Tail **transcribed verbatim**. | ✅ **Recovered** — their code, not ours |
| `FINCH(..., tw_finch=...)` — `tw_finch` is absent from every release of the pinned `finch-clust==0.2.0` | Nothing in the repo permits transcription. TW-FINCH is published (Sarfraz et al., CVPR 2021) with a one-line rule: weight distance by temporal proximity, `d·\|i−j\|`. Applied through FINCH's own `initial_rank` hook. Validated: mean temporal jump = **1.00** vs **6.72** for standard FINCH. | ⚠️ **Reconstructed** — OUR implementation |

Because of the second row, any number is reported as **"DyTo (reconstructed
TW-FINCH)"** and never as "DyTo". `summary.json` carries `finch_variant`,
`paper_faithful: false` and `reconstruction_notes`, so the caveat travels with the
data rather than living only in prose.

**Measured A/B — a null result.** Standard FINCH (`$DYTO_TW_OFF=1`, job 7786513)
vs reconstructed TW-FINCH (job 7786505): **0/8 predictions differ**. The clustering
choice is not observable in the output at n=8 — DyTo's downstream ToMe merge and
its ~25-frame cap absorb it. Two consequences: the reconstruction is *unlikely to
be the load-bearing part* of whatever DyTo scores, and no claim that the temporal
weighting changes results is supportable without a larger paired comparison.

**Config note:** 32 frames, not the paper's 100 — 100 OOMs on a 44 GiB card, and
FINCH selects ~25 representative frames regardless.

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

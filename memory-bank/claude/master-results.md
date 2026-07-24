# Master Results — HuBenchmarks / MotionBench

**Updated** 2026-07-23 · **Benchmark** MotionBench (8,052 samples · 4,018 scoreable · 4,034 NA)
**Standard config** 32 frames · retention 0.15 where the method exposes one · `do_sample=False` · `max_new_tokens=16` · letter-match scoring · NA skipped

**Goal** — 11 methods × 2 backbones (LLaVA-OV-7B, Qwen3-VL-8B) = **22 cells**, each a paper-exact implementation verified to actually engage.

---

## 1. Legend

| Mark | Meaning |
|:---:|---|
| ✅ | Verified: method provably engages (smoke-gated, predictions diverge from baseline) |
| 🟡 | Provisional: number exists but predates the verification harness — **not** gated |
| ❌ | Invalid: silent no-op, crash, or below random. **Not a result.** |
| 🔄 | In flight |
| 📋 | Not yet implemented |

> **Read this before quoting any number.** Every full-run figure below was produced
> **before** today's fixes and carries no `enabled` flag, so none was divergence-gated.
> They are the best available figures but remain **provisional (🟡)** until Wave 2
> re-runs them. Numbers marked ❌ are known-invalid and must not be reported.

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
| **VideoITG** | ⚠️ | Runs, but its infer script emits no `*_params` block, so engagement can't be confirmed |
| **DyTo** | ❌ | **Never ran.** 5 import failures (`LlavaLlamaForCausalLM`). Deferred to last |

---

## 3. Stage 1 — LLaVA-OV-7B (provisional numbers)

| # | Method | Overall | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count | Params |
|:-:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | **DyCoke** 🟡 | **53.36%** | 38.92% | 48.83% | 55.68% | 58.19% | 70.87% | 25.25% | l=3, p=0.7, k=0.7 |
| 2 | **FlashVID** (0.25) 🟡 | **53.36%** | 42.20% | 47.53% | 55.13% | 57.58% | 71.74% | 23.75% | retention=0.25 |
| 3 | **FlashVID** (0.15) 🟡 | **53.29%** | 39.69% | 45.71% | 54.03% | 57.85% | 71.88% | 28.25% | retention=0.15 |
| 4 | **HoliTom** 🟡 | **53.14%** | 40.85% | 49.61% | 52.56% | 57.04% | 71.45% | 27.25% | RETAIN=0.15, T=0.80, k=18, r=0.5 |
| 5 | **MDP3** 🟡 | **53.06%** | 40.46% | 49.61% | 53.48% | 56.77% | 71.59% | 26.50% | pool=32, select=8 |
| 6 | **VideoITG** 🟡 | **52.86%** | 40.08% | 47.01% | 53.66% | 57.58% | 70.14% | 26.75% | grounding: 512 sample, 32 select |
| 7 | **AIM** 🟡 | **52.84%** | 41.43% | 47.79% | 54.03% | 57.10% | 71.88% | 22.25% | bipartite merge + PageRank |
| 8 | **STTM-v2** 🟡 | **51.87%** | 39.11% | 48.83% | 53.66% | 53.59% | 71.16% | 29.25% | layer=2, thresh=0.85 |
| — | *Backbone baseline* | *52.66%* | *40.46%* | *45.19%* | *55.49%* | *57.04%* | *71.16%* | *23.75%* | *plain LLaVA-OV-7B* |
| — | **FastV** 🔄 | — | | | | | | | k=2, r=0.85 (keeps 15%) — Wave 2 |
| — | **VisionZip** 🔄 | — | | | | | | | dominant=54, contextual=10 — Wave 2 |

### ❌ Invalid entries — do not report

| Label | Number | Why it is not a result |
|---|:---:|---|
| "FastV" | 52.66% | `apply_fastv()` was a stub, `enabled: false`. This is the bare backbone |
| "PruneVID (OV port)" | 52.66% | **0/8052** predictions differ from the inert run above. VTP never fired |
| "VisionZip" | 0.00% ×3 | Output-slicing bug (fixed today) discarded every response |
| "VideoITG (simplified)" | 34.89% | Single-stage, no grounding — not the method |
| "FlashVID (qwen15)" | 51.22% | Ran at retention 0.10 with the wrong model class |

---

## 4. Stage 3 — Qwen3-VL-8B

> **Was never runnable.** Every pre-existing env has transformers 4.45, which lacks
> `Qwen3VLForConditionalGeneration`. New env **`qwen3vl`** (torch 2.6.0+cu124,
> transformers 5.14.1) fixes this.
>
> **Frame bug fixed:** `Qwen3VLVideoProcessor` has `do_sample_frames=True, fps=2`, so it
> re-sampled our frames and ignored `--num_frames`. With `do_sample_frames=False`,
> verified `requested=32 given=32`.
>
> **All 8 existing "ports" are fake** — each `eval_<m>.py` is the same baseline script
> applying no method. They need real re-implementation, not submission.

| Method | Overall | Status |
|---|:---:|---|
| **Baseline** | 🔄 | Full run in flight (job 7770357) — Stage-3 Step 0 and the `--vs-baseline` reference for every port |
| FastV | 📋 | Wave 3 #1 — algorithm already resolved on LLaVA-OV |
| DyCoke, HoliTom | 📋 | Wave 3 #2 |
| VisionZip, AIM, MDP3, VideoITG, FlashVID | 📋 | Wave 3 #3 |
| PruneVID, STTM, DyTo | 📋 | Wave 3 #4 — likely **architecture-incompatible** (STTM patches Qwen2 attention; DyTo is Vicuna-bound) |

---

## 5. Other backbones (separate track)

| Method | Backbone | Overall | Status |
|---|---|:---:|---|
| PruneVID | PLLaVA-7B | **44.00%** 🟡 | Real result (`pruning_enabled: True`) |
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

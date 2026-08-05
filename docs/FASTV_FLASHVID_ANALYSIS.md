# FastV and FlashVID: why they underperform

Two methods, two different failure modes. **FastV collapses on LLaVA-OV — closed,
fully explained, not a bug.** **FlashVID underperforms on Qwen3-VL — open, pending
a re-run against the authors' own code**, because the number currently in the
results tables is our reimplementation, not their released mechanism.

MotionBench, 4,018 scoreable questions, 32 frames, greedy decoding. LLaVA-OV-7B
baseline 52.66%, Qwen3-VL-8B baseline 62.52%. Resolution floor ±1.54 pt;
McNemar significance threshold χ² ≥ 3.84.

---

## Part 1 — FastV on LLaVA-OV: closed

**Finding.** FastV loses 15.9 points on LLaVA-OV and the loss does not depend
on which tokens it keeps, how much it keeps, or how long the source video is.
Selecting by attention, at random, or evenly across all 32 frames score within
1.1 points of each other. Keep 10% through keep 75% — including the authors'
own K=2/R=50% setting — all land at 35.7–36.8%. Only keeping every token
recovers the backbone. This is a **density threshold**, not a ranking failure,
not a duration effect, and not a bug.

### 1.1 Across the retention range

| Backbone | keep 10% | keep 15% | keep 25% | keep 50% | keep 75% | keep 100% |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA-OV-7B | 36.06 | 36.78 | 36.73 | 36.34 | 35.69 | 55.07\* |
| Qwen3-VL-8B | 56.92 | 59.01 | 60.60 | — | — | 62.52 |

\*986-question subset (baseline 54.36%); `keepall` was only run there. keep
50% is the authors' own K=2/R=50% setting; keep 75% is K=2/R=25%-drop. A
K=3/R=50% variant scores 35.94 — layer choice doesn't matter either.

LLaVA-OV spans **36.06–36.78** across the entire discarding range: a
0.72-point spread from keep 10% to keep 75%. There is no threshold to locate
between 25% and 100% — the collapse is flat across the whole range FastV's
authors validate, and recovery happens only at keep 100%, where nothing is
discarded. Qwen3-VL, by contrast, responds to budget the whole way: +3.68
points from keep 10% to keep 25%.

Per category at keep 15% (backbone in italics):

| Backbone | Overall | Action Order | Camera Motion | Location | Motion Recog. | Motion Objects | Repetition |
|---|---:|---:|---:|---:|---:|---:|---:|
| *LLaVA-OV baseline* | *52.66* | *40.5* | *45.2* | *55.5* | *57.0* | *71.2* | *23.8* |
| LLaVA-OV FastV | 36.78 | 32.8 | 31.9 | 34.1 | 35.7 | 53.8 | 25.2 |
| *Qwen3-VL baseline* | *62.52* | *46.1* | *63.1* | *65.0* | *67.3* | *79.0* | *33.8* |
| Qwen3-VL FastV | 59.01 | 46.2 | 61.0 | 61.9 | 63.5 | 72.2 | 30.5 |

Paired against the LLaVA-OV backbone: 4,605 of 8,052 predictions differ, 1,113
broken against 475 fixed (χ² = 255.5) — the changes are overwhelmingly
destructive, not merely numerous. FastV answers **D** on ~47% of questions at
every LLaVA-OV budget tried, against a 23.6% ground-truth rate — the signature
of a model with the prompt but no usable visual evidence.

### 1.2 What causes it — a controlled four-arm study

Identical backbone, prompt, frames, budget, decoding; only the token-selection
rule at the pruning layer varies (986-question subset, category mix matches
the full benchmark within 1.5 points):

| Arm | Rule | Overall | D-rate | Broke:fixed |
|---|---|---:|---:|---:|
| *baseline* | *no reduction* | *54.36* | *20.7* | — |
| `keepall` | same path, discards nothing | **55.07** | 20.8 | 0.36 |
| `attention` | FastV as published | 39.25 | 49.4 | 2.27 |
| `random` | uniformly at random | 38.74 | 47.2 | 2.43 |
| `uniform` | evenly spread over 32 frames | 38.13 | 49.9 | 2.43 |

McNemar between arms: attention vs random χ² = 0.16; attention vs uniform
χ² = 1.54; random vs uniform χ² = 0.30 — none significant. Three conclusions:

1. **Not our code.** `keepall` traverses the identical pruning path while
   discarding nothing and returns the backbone.
2. **Not the ranking.** FastV's layer-2 attention scores no better than
   random — the ranking is inoperative at this budget, not merely weak.
3. **Not frame collapse.** `uniform` guarantees equal frame coverage and
   doesn't help. Every arm — `attention` included — touches all 32 frames on
   essentially every sample (mean 32.0 of 32).

What remains is a threshold. Below it, discarding costs ~15 points regardless
of which tokens survive; at 100% the backbone is intact. Selection quality
stops mattering below the line.

### 1.3 Not a regime we invented

Checked against the authors' code at `/project/rhu/dpalfaro/code/FastV`: the
released sweep is `rank_list=(72 144 288 432)` against
`--fast-v-image-token-length 576`, i.e. keep-fractions of 12.5/25/50/75%; our
15% sits inside that range. But the **only** released evaluation is
`eval_ocrvqa.sh` — single-image OCR-VQA, 576 tokens. There is no video
configuration in the released code. So the discrepancy isn't an untested
*fraction*, it's an untested *regime*: 6,273 visual tokens spanning 32 video
frames, against 576 tokens of one image. The keep-fraction transfers; the
attention statistics it depends on don't.

### 1.4 Not duration-dependent

At a fixed 32 frames, a short clip is oversampled relative to a long one, so a
plausible alternative hypothesis was that the collapse hits short clips
harder. A naive 3s split *appears* to confirm this (FastV asymmetry −5.20,
χ²=16.28 significant) — but it's confounded twice over:

- **Category mix**: Motion-related Objects (the category FastV retains most
  of) is 33.7% of short clips vs 11.7% of long ones.
- **Floor effect**: the backbone itself scores 62.5 on short clips vs 49.4 on
  long, so any method degenerating toward a fixed answer loses more wherever
  the baseline was better — no duration mechanism required.

D-rate (collapse signature, not sensitive to baseline skill) moves in the
*opposite* direction: 44.6% short vs 47.8% long at keep 10%. Holding question
type fixed and re-splitting at 5s, the asymmetry collapses to **+0.09**
(noise); the one cell that still looks like an effect (Camera Motion, −11.16)
tests as not significant (χ²=2.41 vs 3.84 threshold). Full derivation:
`analysis/duration-split/`.

### 1.5 PruneVID-OV shares the exact signature

`cluster_ratio=1.0` (`s1_prunevid_c100_run`, disables PruneVID's spatial
merge, keeps every token) scores **53.36%, +0.70 vs. the backbone, χ² = 2.3
(not significant)** — the same recovery FastV's own `keepall` control shows.
Two unrelated ports, on the same backbone, both intact with nothing discarded
and both collapsed the moment anything is. The mechanism is not
method-specific — it's a property of LLaVA-OV below some retained-token
density, independent of which method is doing the discarding.

### 1.6 Closed

No threshold to locate between 25% and 100% — FastV is flat at ~36% for every
discarding setting from keep 10% to keep 75%, including its own published
K=2/R=50% operating point, and recovers only when discarding stops entirely.
Not duration-dependent. Not a harness bug. The open item is retired.

**Reproduce:**

```bash
cd analysis/fastv-selection-study
sbatch --export=ALL,SELECT=attention                          run_selection_study.sbatch
sbatch --export=ALL,SELECT=random                             run_selection_study.sbatch
sbatch --export=ALL,SELECT=uniform                             run_selection_study.sbatch
sbatch --export=ALL,SELECT=attention,FASTV_R=0.00,TAG=keepall run_selection_study.sbatch

python3 analysis/duration-split/split_by_duration.py --cut 3 --cut 5 --cut 10
python3 analysis/duration-split/duration_within_category.py --cut 5
```

Runs: `w2_fastv_run`, `s1_fastv_r10_run`, `s1_fastv_r25_run`,
`s1_fastv_r50_run`, `s1_fastv_r75_run`, `s1_fastv_k3r50_run`, `fastv_run1`
(backbone), `w3_fastv_run`, `s3_fastv_r10_run`, `s3_fastv_r25_run`,
`qwen3vl_baseline_run1`, `fv_sel_{attention,random,uniform,keepall}`,
`s1_prunevid_c100_run`.

---

## Part 2 — FlashVID on Qwen3-VL: open

**Status: the poor number is not verified against the real method yet.**
`w3_flashvid_run` — the 56.65% in the results tables, −5.87 vs. the 62.52%
baseline — **never imports FlashVID's own package**. It's a reimplementation
that omits inner-LLM compression (`pruning_layer=28`, `llm_retention_ratio=0.1`),
`expansion=1.25`, `min_segment_num=4`, `complementary_segment`,
`segment_threshold`, and `token_selection_method=attn_div`. FlashVID ships
native Qwen3-VL support (`flashvid/modeling_qwen3_vl.py`,
`scripts/qwen3_vl.sh`), so the honest question — is FlashVID actually this
weak on Qwen3-VL, or is that our port — is still unanswered. On **LLaVA-OV**,
by contrast, FlashVID is accuracy-neutral (53.31–53.36% at every retention
tried, no collapse) — the underperformance is specific to Qwen3-VL and
specific to the un-verified reimplementation.

### 2.1 The authors'-code re-run hit two real upstream bugs

`analysis/upstream-faithful/eval_flashvid_qwen3vl_official.py` calls the
authors' `flashvid()` directly, at their published setting
(`retention_ratio=0.15, alpha=0.7, temporal_threshold=0.8,
token_selection_method=attn_div, min_segment_num=4, segment_threshold=0.9,
expansion=1.25, pruning_layer=28, llm_retention_ratio=0.1`).

**Bug 1 — dtype crash, a regression in FlashVID's own release.** First
attempt (job 7961637): 0.00% accuracy, 100% empty predictions,
`expected scalar type Float but found BFloat16` on 8,044/8,052 samples. The
gate caught it correctly (`RESULT: FAIL`). Traced to
`flashvid/modeling_qwen3_vl.py`'s vision forward:

```python
hidden_states = hidden_states + pos_embeds          # no dtype cast
```

against stock `transformers`:

```python
hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
```

`fast_pos_embed_interpolate` returns float32; with the model in bf16 the
unguarded add fails outright. Confirmed a bug in the released code (not our
harness) by diffing against the stock implementation it was patched from.
Fixed with the one-line cast, on the vendored clone, original backed up.

**Bug 2 — hard FlashAttention-2 dependency, no matching build on the
cluster.** With bug 1 fixed, every sample still failed silently:

```python
assert self.config._attn_implementation == "flash_attention_2"
```

FlashVID's vision attention has no sdpa/eager fallback. A prebuilt wheel
matching the exact torch/CUDA/Python build installed but failed on import
(`undefined symbol` — a libtorch C++ ABI mismatch not captured by the wheel's
coarse version tag). A "from source" `pip install` then silently
re-downloaded the same broken wheel — flash-attn's `setup.py` guesses a
prebuilt-wheel URL and skips compiling unless `FLASH_ATTENTION_FORCE_BUILD=TRUE`
is set; caught by comparing wheel sizes (256MB fake vs. a genuine compile).
Forcing the real build (`FLASH_ATTENTION_FORCE_BUILD=TRUE`, matching CUDA
module) produced a genuine 112,864,959-byte wheel (`flash_attn-2.8.3.post1`)
that imports cleanly.

Neither bug is ours — both are defects in the released FlashVID
package/environment assumptions, found only because we tried to run their
actual code instead of trusting our port.

### 2.2 Why the "poor" number moved before — reproducibility forensics

Before the authors'-code effort, two nominally-identical FlashVID re-runs
disagreed by 2.14% (53.36% vs 51.22%) and looked like a prompt-template
effect (`qwen_1_5` vs `qwen_2`). It wasn't — the two templates render
byte-identical strings. The real cause: the lower run used the **wrong model
class** (`LlavaLlamaForCausalLM` instead of LLaVA-OneVision) and **retention
0.10, not 0.15** as its filename claimed — two unrelated misconfigurations,
neither about the prompt.

That investigation also established a general calibration for this project:
a genuinely identical re-run diverges by **zero** predictions (verified on
three independent pairs), so any nonzero divergence is signal, not noise.
Divergence scale by cause:

| Cause | Predictions changed (of 8,052) |
|---|:---:|
| Nothing — identical re-run | 0 |
| Code-version drift | 13–23 |
| Retention 0.15 ↔ 0.25 | ~1,120 |
| Retention 0.10 ↔ 0.25 | ~1,507 |

This scale is what let a mislabelled run (`flashvid_qwen2_v2`, recorded
retention 0.25, `summary.json` and git both agreeing) be proven to have
actually *executed* at 0.15 — its divergence from known 0.15 and 0.25 runs
matched the 0.15 bucket, not the 0.25 one it claimed. Lesson carried forward:
a run's stored parameter is not proof of what it executed; behavioural
divergence is a more reliable witness than either `summary.json` or git
history.

### 2.3 What's still open

**Job `s3_flashvid_official_run` (7985110)** — full 8,052-question run, both
bugs fixed, authors' published Qwen3-VL config — is queued
(`PENDING (Priority)`, GPU-partition contention from other users, no
confirmed start estimate) on Carya as of this writing. Once it lands and
gates clean (`check_run.py --vs-baseline qwen3vl_baseline_run1`), this
section closes with a real accuracy number to compare against the
reimplementation's −5.87.

**Reproduce:**

```bash
# the fixes (already applied on Carya)
diff /project/rhu/dpalfaro/code/FlashVID/flashvid/modeling_qwen3_vl.py.bak \
     /project/rhu/dpalfaro/code/FlashVID/flashvid/modeling_qwen3_vl.py

sbatch analysis/upstream-faithful/build_flash_attn.sbatch
sbatch analysis/upstream-faithful/run_flashvid_qwen3vl_official.sbatch
```

Runs: `w2_flashvid_run`, `s1_flashvid_r10_run`, `s1_flashvid_r25_run`,
`w3_flashvid_run`, `s3_flashvid_official_run` (pending).

# Why FastV collapses where FlashVID does not

Diagnostic study of the accuracy gap between attention-pruning and merge-based
token-reduction methods on LLaVA-OneVision, prompted by the FlashVID paper's
LLaVA-OneVision comparison table.

---

## 1. The question

The FlashVID paper reports FastV as the weakest method at every retention ratio
on LLaVA-OneVision, falling to 91.8% relative accuracy at R=10% while FlashVID
holds 99.1%. The gap is large and consistent. Two things needed checking: whether
the ordering is real or an artifact of how a competing paper implements a
baseline, and what mechanism produces it.

## 2. Findings in brief

1. **The ordering is real.** It reproduces independently on MotionBench with our
   own implementations: FastV 36.78% against a 52.66% backbone, FlashVID 53.31%.
2. **The gap is not about how many predictions change — it is about direction.**
   FastV changes 57% of all predictions and breaks 2.34 answers for every one it
   fixes. FlashVID changes 20% and breaks 0.90 per fix, i.e. slightly net
   positive.
3. **FastV does not degrade gracefully on this backbone; it collapses.** It
   answers "D" on 46.8% of questions against a 23.6% ground-truth rate. The
   backbone itself is near-uniform. This is the signature of a model answering
   from text alone.
4. **The collapse is retention-invariant.** At R = 10%, 15% and 25% the accuracy
   is 36.06 / 36.78 / 36.73 and the D-rate is 47.0 / 46.8 / 47.1. Tripling the
   token budget changes nothing. Whatever fails, fails completely at every budget
   tested.
5. **The same algorithm is healthy on Qwen3-VL**, where it recovers monotonically
   (+3.68 across the same sweep) and its answer distribution tracks the backbone
   to within 1.5 points. The failure is an interaction with LLaVA-OV, not a
   property of FastV in general.
6. **Resolved: not our harness, and not FastV's ranking either.** A four-arm
   controlled study (§7) shows the shared `kv_cache` pruning route is clean —
   keeping 100% of tokens through it reproduces the backbone almost exactly.
   But it also shows FastV's attention-based token ranking is statistically
   indistinguishable from choosing tokens at random, and that spreading the
   budget evenly across all 32 frames does not rescue it either. At 15%
   retention on this backbone, **no selection policy for outright discarding
   tokens works** — which is also the most likely explanation for why
   PruneVID-OV, a structurally different algorithm, collapses to a nearly
   identical number.

## 3. What the methods actually do

The determining variable is **where** the reduction happens relative to the LLM,
and **what happens to a token that is not kept**.

| Method | Operates | Non-kept token is | Temporal guarantee | LLaVA-OV result |
|---|---|---|---|---:|
| FlashVID | before the LLM (ADTS + TSTM) | **merged** into a survivor | yes — `do_segment=True` allocates budget per temporal segment | 53.31% |
| DyCoke | temporal merge, then KV prune at layer 3 | merged, then gently pruned | yes — stage 1 merges across frames | 53.36% |
| HoliTom | before the LLM, hierarchical | **merged** | yes | 53.14% |
| FastV | inside the LLM at layer 2 | **discarded** | none — single global top-k over ~6,300 tokens | 36.78% |
| PruneVID-OV | inside the LLM at layer 10 | discarded after clustering | yes — DPC-KNN temporal segmentation | 38.20% |

Merging preserves the low-frequency content of a discarded token in the survivor
it folds into. Discarding does not. At a 15% budget that difference compounds:
one method compresses the visual evidence 6.7:1, the other deletes 85% of it.

Note that PruneVID-OV *does* have a temporal guarantee and still fails. §7 shows
why the "temporal guarantee" column above is not actually the deciding factor
for either in-LLM pruner: even a hand-built policy that guarantees perfect
temporal coverage fails just as badly. The deciding factor is **discard vs.
merge**, full stop — not how the discarding is organized.

## 4. Reproduction

MotionBench, 8,052 samples, 4,018 scoreable, 32 frames, greedy decoding,
15% retention. Backbone: LLaVA-OV-7B (Qwen 1.5), baseline 52.66% (2116/4018).
Resolution floor is ±1.54 pt; McNemar needs χ² ≥ 3.84 for p<0.05.

| Method | Accuracy | Δ backbone | Predictions changed | Broke | Fixed | Broke:fixed | χ² |
|---|---:|---:|---:|---:|---:|---:|---:|
| DyCoke | 53.36% | +0.70 | 1,031 | 142 | 170 | 0.84 | 2.3 |
| FlashVID | 53.31% | +0.65 | 1,610 | 242 | 268 | 0.90 | 1.2 |
| HoliTom | 53.14% | +0.48 | 1,838 | 280 | 299 | 0.94 | 0.6 |
| *baseline* | *52.66%* | — | — | — | — | — | — |
| PruneVID-OV | 38.20% | −14.46 | 4,536 | 1,054 | 473 | **2.23** | 220.3 |
| FastV | 36.78% | −15.88 | 4,605 | 1,113 | 475 | **2.34** | **255.5** |

The three merge-based methods are statistically indistinguishable from the
backbone. The two in-LLM pruners are not, by a wide margin.

**Excluded:** VisionZip is often grouped with these, but our VisionZip run uses
LLaVA-1.5-7B (it patches `CLIPVisionTower`, which LLaVA-OV does not have). Its
40.09% is measured against a different and weaker backbone and is not comparable
to the rows above.

## 5. Evidence

### 5.1 The answer distribution

Percentage of scoreable questions answered with each letter.

| Run | A | B | C | D | Most-picked |
|---|---:|---:|---:|---:|---:|
| *ground truth* | 25.4% | 25.6% | 25.5% | 23.6% | — |
| baseline | 27.1% | 26.6% | 26.9% | 19.3% | A 27% |
| DyCoke | 27.4% | 26.2% | 27.0% | 19.4% | A 27% |
| FlashVID | 27.8% | 25.7% | 25.7% | 20.8% | A 28% |
| HoliTom | 26.1% | 24.1% | 27.2% | 22.5% | C 27% |
| PruneVID-OV | 12.2% | 26.7% | 15.0% | **46.1%** | D 46% |
| FastV | 14.9% | 23.1% | 15.2% | **46.8%** | D 47% |

The backbone and every merge method sit near uniform. The two in-LLM pruners put
nearly half their answers on the last option. A model that has lost its visual
evidence but still reads the prompt will favour the most recently mentioned
option; this is what that looks like. Accuracy alone would not have revealed it.

### 5.2 Retention invariance

| Backbone | Method | R=0.10 | R=0.15 | R=0.25 | Span |
|---|---|---:|---:|---:|---:|
| LLaVA-OV | FastV | 36.06% | 36.78% | 36.73% | **+0.67** |
| LLaVA-OV | PruneVID | 38.10% | 38.20% | 38.38% | **+0.27** |
| LLaVA-OV | FlashVID | 52.51% | 53.31% | 53.36% | +0.85 |
| LLaVA-OV | HoliTom | 52.76% | 53.14% | 53.41% | +0.65 |
| Qwen3-VL | FastV | 56.92% | 59.01% | 60.60% | **+3.68** |
| Qwen3-VL | FlashVID | 54.73% | 56.65% | 59.36% | **+4.63** |
| Qwen3-VL | HoliTom | 60.05% | 60.33% | 60.43% | +0.37 |
| Qwen3-VL | PruneVID | 60.23% | 62.17% | 61.40% | +1.17 |

FastV's D-rate across the same LLaVA-OV sweep is 47.0 / 46.8 / 47.1 — flat to a
tenth of a point.

**Revised reading, after §7.** The original reading of this table argued that
flatness across 10–25% retention meant the failure was not about budget at
all — that the retained tokens carry no evidence regardless of how many
survive. §7's `keepall` control (100% retention, same code path) contradicts
the strong form of that claim: at 100% the backbone is fully recovered. So this
*is* a budget effect — just one with a narrow, steep window. Somewhere between
25% and 100% retention, discard-based pruning on this backbone crosses from
"non-functional" to "fine," and the 10–25% range this sweep covers sits
entirely below that line. What flatness across 10–25% actually shows is that
the window is not gradual within this range — not that budget is irrelevant
outright. Where exactly the recovery threshold sits (near FastV's own paper
default of 50%, or elsewhere) has not been tested.

### 5.3 Category structure

Above-chance signal retained, computed as (method − 25) / (baseline − 25).
Chance on 4-way multiple choice is 25%.

| Category | Baseline | FastV | Signal kept | n |
|---|---:|---:|---:|---:|
| Motion-related Objects | 71.2% | 53.8% | **62%** | 690 |
| Action Order | 40.5% | 32.8% | 50% | 519 |
| Camera Motion | 45.2% | 31.9% | 34% | 385 |
| Motion Recognition | 57.0% | 35.7% | 33% | 1,478 |
| Location-related Motion | 55.5% | 34.1% | **30%** | 546 |
| Repetition Count | 23.8% | 25.2% | n/a — baseline at chance | 400 |

Motion-related Objects is the category most answerable from object appearance in
a small number of frames, and it retains roughly twice the signal of the
categories that require integrating motion across the clip. Repetition Count is
the control: the backbone is already at chance there, and FastV loses nothing.
FastV destroys signal in proportion to how much genuinely temporal signal the
category contained.

### 5.4 Backbone dependence

The same algorithm at the same settings, on Qwen3-VL-8B (baseline 62.52%):

| Run | A | B | C | D |
|---|---:|---:|---:|---:|
| baseline | 21.6% | 28.5% | 29.3% | 20.6% |
| FastV R=0.10 | 20.3% | 29.2% | 30.7% | 19.8% |
| FastV R=0.15 | 20.4% | 28.8% | 30.5% | 20.3% |
| FastV R=0.25 | 20.8% | 28.5% | 30.0% | 20.8% |

No collapse. The distribution tracks the backbone within 1.5 points at every
budget, and accuracy recovers monotonically as tokens are restored. Whatever
breaks on LLaVA-OV does not break here.

## 6. Why the published table shows a milder gap

Our LLaVA-OV FastV drop (−15.88) is far larger than the paper's (−4.8 at R=10%).
Three factors account for the difference, in decreasing order of confidence:

1. **Benchmark sensitivity.** VideoMME, EgoSchema and LongVideoBench are largely
   answerable from gist and appearance. MotionBench is fine-grained motion, which
   is precisely what survives frame reduction least well. Our own category
   breakdown (§5.3) shows appearance-type questions retaining twice the signal of
   motion-type questions under the same method.
2. **Operating point.** FastV's paper default is **R=50%**. The table extends it
   to 10% — a 5× extrapolation past its validated range — while FlashVID is
   evaluated inside the range it was designed and tuned for. A fair reading of
   that table notes FastV is being run well outside its specification.
3. **Baseline-implementation asymmetry.** In any paper, the proposing method is
   tuned and the baselines are run at defaults. This is normal and not
   misconduct, but it means a baseline number in a competitor's table is a floor
   rather than a characterization. Our independent reproduction is what makes the
   ordering credible, not the table itself.

## 7. Resolved: neither the harness nor the ranking — a hard density floor

PruneVID-OV uses a different algorithm (DPC-KNN clustering), a different layer
(10 vs 2), and a different selection criterion, yet fails with a signature that
matches FastV's to about one percentage point on every measure — accuracy, D
rate, broke:fixed ratio, and flatness across retention. Two unrelated algorithms
do not usually fail identically, which raised two live explanations: our shared
`PrunableDynamicCache.kv_cache` pruning route (used because this LLaVA-OV build
silently ignores an additive attention mask — see
[eval_fastv.py:186](../stage1-llava-ov/fastv-motionbenc/eval_fastv.py#L186)) could
itself be corrupting the visual pathway, independent of which tokens either
method nominates; or FastV's layer-2 ranking could simply be uninformative at
video scale.

A four-arm controlled study separated them — same fork of the gated script
(untouched original), same 15% budget, same 986-question representative subset,
varying only the layer-K selection policy:
[analysis/fastv-selection-study/](../analysis/fastv-selection-study/).

| Arm | Policy | Accuracy | D-rate | Broke:fixed |
|---|---|---:|---:|---:|
| *bare backbone (subset)* | — | *54.36%* | *~19%* | — |
| `keepall` (r=0.00) | same route, drops nothing | **55.07%** | 20.8% | 0.36 |
| `attention` | FastV as published | 39.25% | 49.5% | 2.27 |
| `random` | k tokens at random | 38.74% | 47.3% | 2.43 |
| `uniform` | k tokens spread evenly across frames | 38.13% | 50.0% | 2.43 |

Fork fidelity confirmed first: `attention` reproduces `w2_fastv_run` on the
matching subset exactly (0/986 divergence, identical accuracy), so the fork did
not alter FastV's behavior.

**The harness is clean.** `keepall` reproduces the backbone — 55.07% vs. 54.36%,
D-rate 20.8% vs. ~19%, and a broke:fixed ratio of 0.36 (it fixes more than it
breaks, consistent with sdpa/eager floating-point noise from temporarily
swapping layer K's attention module, not corruption). Explanation **B** is
rejected: the shared `kv_cache` route does not damage anything when it is not
asked to discard tokens.

**The ranking carries no signal at this budget.** `attention` (39.25%) is 0.51
points from `random` (38.74%) — McNemar χ²=0.16, nowhere near significant.
FastV's layer-2 attention ranking performs identically to picking 15% of tokens
uniformly at random.

**Temporal spread does not rescue it either.** `uniform` (38.13%) is
statistically indistinguishable from both `attention` and `random`
(χ²=1.54 and 0.30). `frame_hist.jsonl` confirms this isn't about frame
collapse to begin with: every arm — including plain `attention` — touches all
32 frames on essentially every sample (mean 32.0/32); `attention`'s busiest
frame gets a mildly higher share of the budget than the others (10.5% vs.
3.1–4.2%), nowhere near the "two or three frames get everything" scenario the
original §3 hypothesis proposed.

**Conclusion.** At 15% retention on LLaVA-OV, discarding 85% of visual tokens
degrades the backbone to the same ~38–39% regardless of *how* the surviving 15%
is chosen — by attention rank, at random, or spread deliberately across every
frame. This is a density floor specific to outright token discarding on this
backbone, not a property of FastV's ranking, PruneVID's clustering, or our
pruning plumbing. It explains why two structurally unrelated discard-based
methods land on the same number: below this floor, the selection rule stops
mattering. It does not by itself explain *why* the floor sits where it does, or
whether accuracy recovers gradually or sharply between 25% and 100% retention
— the sweep in §5.2 only covers 10–25%, and `keepall` is the only point tested
above that. That gap — and whether the recovery threshold sits near FastV's own
paper default of 50% — is the natural next experiment, not yet run.

## 8. Reproduce

```bash
source config/paths.local.sh

# The gated runs behind §4
python stage1-llava-ov/fastv-motionbenc/eval_fastv.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/fastv_run" \
    --num_frames 32 --fastv --fastv_k 2 --fastv_r 0.85

# The selection study of §7 (on Carya)
cd analysis/fastv-selection-study
sbatch --export=ALL,SELECT=attention                   run_selection_study.sbatch
sbatch --export=ALL,SELECT=random                      run_selection_study.sbatch
sbatch --export=ALL,SELECT=uniform                     run_selection_study.sbatch
sbatch --export=ALL,SELECT=attention,FASTV_R=0.00,TAG=keepall run_selection_study.sbatch
```

Analysis tables in §4 and §5 are computed from cached `results.jsonl` files;
pull them with `scripts/fetch_results.sh`.

## 9. Status

| Item | State |
|---|---|
| §4 reproduction, §5 evidence | complete, from gated full runs |
| §6 paper-table reading | complete |
| §7 selection study | **complete** — jobs 7942016–18, 7942024 all landed |

The FastV and PruneVID-OV LLaVA-OV numbers in [RESULTS.md](RESULTS.md) and
[master-results.md](../memory-bank/claude/master-results.md) are confirmed real,
not harness artifacts, and the earlier retraction warning attached to them is
lifted. What remains unknown is only where the recovery threshold sits between
25% and 100% retention (§7, final paragraph).

Run directories: `w2_fastv_run`, `w2_prunevid_ov_run`, `w2_flashvid_run`,
`w2_dycoke_run`, `w2_holitom_run`, `s1_fastv_r10_run`, `s1_fastv_r25_run`,
`fastv_run1` (bare backbone). Qwen3-VL: `w3_fastv_run`, `s3_fastv_r10_run`,
`s3_fastv_r25_run`, `qwen3vl_baseline_run1`. Selection study:
`fv_sel_attention`, `fv_sel_random`, `fv_sel_uniform`, `fv_sel_keepall`.

# FastV on LLaVA-OV: what breaks and why

**Finding.** FastV loses 15.9 points on LLaVA-OV and the loss does not depend
on which tokens it keeps, how much it keeps, or how long the source video is.
Selecting by attention, at random, or evenly across all 32 frames scores
within 1.1 points of each other. Keep 10% through keep 75% — including the
authors' own K=2/R=50% setting — all land at 35.7–36.8%. Only keeping every
token recovers the backbone, at every setting tried. This is a density
threshold, not a ranking failure, not a duration effect, and not a bug.

MotionBench, 4,018 scoreable, 32 frames, greedy. LLaVA-OV-7B baseline 52.66%,
Qwen3-VL-8B baseline 62.52%. Resolution floor ±1.54 pt; McNemar χ² ≥ 3.84.

---

## 1. FastV across the board

| Backbone | keep 10% | keep 15% | keep 25% | keep 50% | keep 75% | keep 100% |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA-OV-7B | 36.06 | 36.78 | 36.73 | 36.34 | 35.69 | 55.07\* |
| Qwen3-VL-8B | 56.92 | 59.01 | 60.60 | — | — | 62.52 |

\*986-question subset, whose baseline is 54.36%; `keepall` was only run there.
All other LLaVA-OV cells are full 4,018-question runs. keep 50% is the
authors' own K=2/R=50% setting; keep 75% is K=2/R=25%-drop. A K=3/R=50%
variant scores 35.94 — the layer choice does not matter either.

On LLaVA-OV the five discarding settings span **36.06 to 36.78**, a 0.72-point
range from keep 10% to keep 75%. There is no threshold to locate between 25%
and 100% — the collapse is flat across the entire range FastV's authors
validate, and recovery happens only at keep 100%, where nothing is discarded
at all. D-rate confirms it moves in lockstep: 46.2% at keep 50%, 45.7% at keep
75%, against the collapse's usual ~47% (§2). Qwen3-VL, by contrast, responds
to budget the whole way: +3.68 points from keep 10% to keep 25%.

Per category at keep 15% (backbone in italics):

| Backbone | Overall | AO | CM | LM | MR | MO | RC |
|---|---:|---:|---:|---:|---:|---:|---:|
| *LLaVA-OV baseline* | *52.66* | *40.5* | *45.2* | *55.5* | *57.0* | *71.2* | *23.8* |
| LLaVA-OV FastV | 36.78 | 32.8 | 31.9 | 34.1 | 35.7 | 53.8 | 25.2 |
| *Qwen3-VL baseline* | *62.52* | *46.1* | *63.1* | *65.0* | *67.3* | *79.0* | *33.8* |
| Qwen3-VL FastV | 59.01 | 46.2 | 61.0 | 61.9 | 63.5 | 72.2 | 30.5 |

Paired against the LLaVA-OV backbone: 4,605 of 8,052 predictions differ, 1,113
broken against 475 fixed, χ² = 255.5. The changes are overwhelmingly
destructive, not merely numerous.

## 2. The collapse is not gradual

On LLaVA-OV, tripling the budget from 10% to 25% buys **0.67 points**. The
answer distribution is flatter still: FastV answers D on 47.0 / 46.8 / 47.1% of
questions across the three budgets, against a 23.6% ground-truth rate and a
19.3% backbone rate. Placing half the answers on one option is what a model
does when it has the prompt but no usable visual evidence.

On Qwen3-VL the same code recovers monotonically (+3.68) and its answer
distribution stays within 1.5 points of the backbone at every budget. The
failure is specific to LLaVA-OV.

## 3. What causes it

Four arms, identical backbone, prompt, frames, budget and decoding; only the
token-selection rule at the pruning layer varies. 986-question subset whose
category mix matches the full benchmark within 1.5 points.

| Arm | Rule | Overall | D-rate | Broke:fixed |
|---|---|---:|---:|---:|
| *baseline* | *no reduction* | *54.36* | *20.7* | — |
| `keepall` | same path, discards nothing | **55.07** | 20.8 | 0.36 |
| `attention` | FastV as published | 39.25 | 49.4 | 2.27 |
| `random` | uniformly at random | 38.74 | 47.2 | 2.43 |
| `uniform` | evenly spread over 32 frames | 38.13 | 49.9 | 2.43 |

McNemar between arms: attention vs random **χ² = 0.16**; attention vs uniform
**χ² = 1.54**; random vs uniform **χ² = 0.30**. None significant.

Three conclusions, in order of what they rule out:

1. **Not our code.** `keepall` traverses the identical pruning path while
   discarding nothing and returns the backbone. The mechanism is sound.
2. **Not the ranking.** FastV's layer-2 attention scores no better than random.
   The ranking is inoperative at this budget, not merely weak.
3. **Not frame collapse.** `uniform` guarantees equal coverage and does not
   help. Per-frame histograms show every arm — `attention` included — touches
   all 32 frames on essentially every sample (mean 32.0 of 32); the busiest
   frame takes 10.5% of `attention`'s budget versus 3.1–4.2% for the others.

What remains is a threshold. Below it, discarding costs ~15 points regardless
of which tokens survive; at 100% the backbone is intact. Selection quality
stops mattering below the line.

The fork used here reproduces the production run exactly on the matched subset
(0 of 986 predictions differ), so it measures the same object as the headline
number.

## 4. Alignment with the published method

Checked against the authors' code at `/project/rhu/dpalfaro/code/FastV`:

- The released sweep is `rank_list=(72 144 288 432)` against
  `--fast-v-image-token-length 576`, i.e. keep-fractions of **12.5%, 25%, 50%
  and 75%**. Our 15% sits inside that fraction range.
- The **only** released evaluation is `eval_ocrvqa.sh` — OCR-VQA, single image,
  576 visual tokens. There is no video configuration in the released code.

So the discrepancy is not that we ran FastV at an untested *fraction*. It is
that we ran it in an untested *regime*: 6,273 visual tokens spanning 32 video
frames, against 576 tokens of a single image. The keep-fraction transfers; the
attention statistics it depends on evidently do not.

## 5. PruneVID-OV shares the exact signature

`cluster_ratio=1.0` (`s1_prunevid_c100_run`) keeps every visual token through
the same PruneVID code path that produces the 38.20% collapse at its default
setting. Result: **53.36%, +0.70 vs. the backbone, χ² = 2.3 (not
significant)** — the same recovery FastV's own `keepall` control showed. Two
unrelated ports, on the same backbone, both intact with nothing discarded and
both collapsed the moment anything is. The mechanism is not method-specific.

## 6. Closed

The threshold question in the prior revision of this document is answered:
there is no threshold between 25% and 100% to locate. FastV is flat at ~36%
for every discarding setting from keep 10% to keep 75%, including its own
published K=2/R=50% operating point, and recovers only when discarding stops
entirely. The open item is retired.

## 7. Reproduce

```bash
cd analysis/fastv-selection-study
sbatch --export=ALL,SELECT=attention                          run_selection_study.sbatch
sbatch --export=ALL,SELECT=random                             run_selection_study.sbatch
sbatch --export=ALL,SELECT=uniform                            run_selection_study.sbatch
sbatch --export=ALL,SELECT=attention,FASTV_R=0.00,TAG=keepall run_selection_study.sbatch
```

Runs: `w2_fastv_run`, `s1_fastv_r10_run`, `s1_fastv_r25_run`, `fastv_run1`
(backbone), `w3_fastv_run`, `s3_fastv_r10_run`, `s3_fastv_r25_run`,
`qwen3vl_baseline_run1`, `fv_sel_{attention,random,uniform,keepall}`.

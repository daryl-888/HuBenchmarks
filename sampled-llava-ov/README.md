# Sampled LLaVA-OV study — answer distribution

**Separate from the gated results. Never merge these numbers into
`master-results.md`.**

## What this is

Eight runs of the LLaVA-OV-7B baseline and its 7 gated methods at the standard
32 frames / 0.15 retention, but with **stochastic decoding** instead of greedy:

    temperature = 0.7,  top_p = 0.9,  seed = 0

The question is not accuracy — it is the **A/B/C/D answer distribution**. A
method whose answers collapse onto one letter is leaning on a default rather
than discriminating, and an accuracy figure alone hides that.

## Why it lives in its own folder

Every script here is a **verbatim copy** of its `stage1-llava-ov/` original with
exactly one behavioural change (greedy → sampled). The originals are untouched
and still produce the gated numbers everything else depends on.

Sampled runs **cannot pass the divergence gate**. That gate works because greedy
decoding is deterministic, so "0 predictions differ from baseline" proves a
method never engaged. Under sampling two runs differ by chance, and the signal
is destroyed. These runs are therefore *descriptive*, not verification.

## Running

```bash
# submit (already done: jobs 7904478-7904485)
for f in run_*_sampled.sbatch; do sbatch $f; done

# once finished: pull results, then build the chart + table
scripts/fetch_results.sh
python3 sampled-llava-ov/build_distribution.py --local
```

Outputs `docs/SAMPLED_DISTRIBUTION.md` (counts + percentages) and
`docs/figures/answer_distribution_llava_ov.png`.

Each script also accepts `--greedy` to reproduce the original behaviour, which
is the cleanest A/B against the gated run.

## What to look for

Ground truth on MotionBench is near-uniform (~25% each). The table flags any
method where one letter exceeds **40%** of answers with ⚠️.

A smoke test of the chart against the *existing greedy* runs already shows the
effect this study is designed to surface:

| Method | A | B | C | D | most-picked |
|---|:---:|:---:|:---:|:---:|:---:|
| *ground truth* | *25.4%* | *25.6%* | *25.5%* | *23.6%* | — |
| baseline | 27.1% | 26.6% | 26.9% | 19.3% | A 27% |
| DyCoke | 20.9% | 28.9% | 29.5% | 20.7% | C 30% |
| FastV | 14.9% | 23.1% | 15.2% | 46.8% | D 47% ⚠️ |
| PruneVID | 12.2% | 26.7% | 15.0% | 46.1% | D 46% ⚠️ |

FastV and PruneVID — the two worst LLaVA-OV scores (36.78%, 38.20%) — both
collapse to ~47% "D". That is the mechanism behind the low accuracy, not just a
symptom of it.

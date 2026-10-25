# FastV and FlashVID: why they underperform

MotionBench, 32 frames, greedy decoding. LLaVA-OV-7B baseline 52.66%, Qwen3-VL-8B baseline 62.52%.

## FastV on LLaVA-OV — closed, density threshold, not a bug

- Loses 15.9 pts (52.66% → 36.78% at keep 15%). Flat 36.06–36.78% from keep
  10% to keep 75%, including the authors' own K=2/R=50% setting (36.34%).
  Recovers only at keep 100% (nothing discarded). Same code on Qwen3-VL
  scales normally with budget — not a harness bug.
- Four-arm test: attention-ranked selection ≈ random (χ²=0.16) ≈ uniform
  spread (χ²=1.54). `keepall` reproduces the backbone exactly. The ranking
  is inoperative at this budget, not just weak, and the mechanism is sound.
- Not duration: naive split looks significant, vanishes (+0.09) once
  category mix and floor effect are controlled for.
- Not method-specific: PruneVID-OV shows the identical collapse/recovery
  signature independently.
- Authors never tested video: their only eval is static-image OCR-VQA, 576
  fixed tokens, no video code anywhere in their repo. We ran 32 frames,
  6,273 tokens. Their own best setting collapses too once transplanted into
  video — it's the regime, not our configuration.

Reproduce: `analysis/fastv-selection-study/`, `analysis/duration-split/`.

## FlashVID on Qwen3-VL — closed: the reimplementation understated it

- The old reported 56.65% (−5.87 vs. baseline) never imported FlashVID's own
  package — a reimplementation missing inner-LLM compression and other
  published parameters. Invalid as a measurement of the method.
- The **authors' actual code, run for real** (`s3_flashvid_official_run`, job
  7987340): **59.36%** (−3.16 vs. the 62.52% baseline) — 2.71 points better
  than the reimplementation suggested. Gate: PASS (2,069/8,052 predictions
  differ from baseline, accuracy in-band, params confirmed live).
- Getting there took three fixes: a dtype-cast bug in their release, a
  FlashAttention-2 build issue, and a `cache_position` crash from a
  `transformers` version mismatch (they pin 4.57.3; the shared env ran
  5.14.1) — fixed via a dedicated pinned env.
- On LLaVA-OV, FlashVID holds steady across the authors' full published
  sweep: 53.31% (0.15) / 53.71% (0.20, χ²=3.91, marginally significant) /
  53.36% (0.25) against a 52.66% baseline — never worse, and at 0.20 a small
  real gain rather than pure noise.

Reproduce: `analysis/upstream-faithful/run_flashvid_qwen3vl_official_v2.sbatch`.

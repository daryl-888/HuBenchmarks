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

## FlashVID on Qwen3-VL — unverified, authors'-code re-run in flight

- The reported 56.65% (−5.87 vs. baseline) never imports FlashVID's own
  package — it's a reimplementation missing inner-LLM compression and other
  published parameters. Invalid as a measurement of the method.
- The LLaVA-OV number (53.31–53.36%, accuracy-neutral) *does* call FlashVID's
  real function with the authors' own default parameters — that one's valid.
- Running the authors' actual Qwen3-VL code took three fixes: a dtype-cast
  bug in their release, a FlashAttention-2 build issue, and a
  `cache_position` crash from a `transformers` version mismatch (they pin
  4.57.3; the shared env ran 5.14.1).
- Debug run against the fixed env now produces real predictions. Full run
  in flight (`s3_flashvid_official_run`, job 7987340) — closes once gated
  against `qwen3vl_baseline_run1`.

Reproduce: `analysis/upstream-faithful/run_flashvid_qwen3vl_official_v2.sbatch`.

# FastV and FlashVID: why they underperform

**FastV collapses on LLaVA-OV — closed, not a bug.** **FlashVID's Qwen3-VL
number is unverified — the authors'-code re-run is in flight.**

MotionBench, 4,018 scoreable questions, 32 frames, greedy decoding. LLaVA-OV-7B
baseline 52.66%, Qwen3-VL-8B baseline 62.52%. Resolution floor ±1.54 pt.

---

## FastV on LLaVA-OV: closed

FastV loses 15.9 points on LLaVA-OV (52.66% → 36.78% at keep 15%). It's a
**density threshold**, not a ranking failure, not duration, not a bug:

- **Flat across the whole retention range.** Keep 10% through keep 75% —
  including the authors' own K=2/R=50% setting (36.34%) — all land at
  36.06–36.78%. Only keep 100% (nothing discarded) recovers the backbone. The
  same code on Qwen3-VL responds normally to budget (+3.68 from keep 10% to
  25%), so this isn't a harness issue.
- **Not the ranking.** A four-arm test (identical backbone/prompt/budget,
  only the selection rule varies) found attention-ranked selection scores no
  better than random (39.25% vs 38.74%, χ²=0.16) or spreading tokens evenly
  across all 32 frames (38.13%, χ²=1.54). `keepall` (same code, discards
  nothing) reproduces the backbone exactly (55.07% vs. 54.36% baseline on
  that subset) — so the mechanism itself isn't broken, and the ranking is
  genuinely inoperative at this budget, not just weak.
- **Not frame collapse.** Every arm, including plain random, touches all 32
  frames on ~every sample.
- **Not duration.** A naive short/long split looks significant but is
  entirely explained by category mix and a floor effect (the backbone itself
  scores 13 points higher on short clips); controlling for question type
  drops the effect to +0.09 (noise). Details: `analysis/duration-split/`.
- **Not method-specific.** PruneVID-OV — an unrelated method — shows the
  identical signature: collapses when discarding anything, recovers
  (53.36%, not significant) when `cluster_ratio=1.0` keeps everything. This
  is a property of LLaVA-OV below some token-density floor, not a bug in
  either port.

**Never tested in this regime.** FastV's released repo has one eval script —
static-image OCR-VQA on LLaVA-v1.5-7B, 576 fixed CLIP tokens, hardcoded
prompt offsets. No video code exists anywhere in their source. We ran
LLaVA-OneVision (SigLIP, not CLIP), 32 frames, 6,273 tokens (11× their
count), motion questions instead of text-reading. The keep-*fraction*
transfers; the regime it was validated in doesn't — and their own
most-published setting collapses identically once transplanted into video,
so it's the regime, not our choice of setting.

**Reproduce:** `analysis/fastv-selection-study/run_selection_study.sbatch`
(arms: `attention`, `random`, `uniform`, `keepall`);
`analysis/duration-split/`. Runs: `w2_fastv_run`, `s1_fastv_r{10,25,50,75}_run`,
`fastv_run1` (backbone), `s1_prunevid_c100_run`.

---

## FlashVID on Qwen3-VL: authors'-code re-run in flight

The 56.65% in the results tables (−5.87 vs. baseline) is `w3_flashvid_run`,
which **never imports FlashVID's package** — it's a reimplementation missing
inner-LLM compression and several other published parameters. Whether
FlashVID is actually this weak on Qwen3-VL, or that's our port, is unverified.
(On **LLaVA-OV**, by contrast, our port does call FlashVID's real function
with the authors' own default parameters and fails loudly if misconfigured —
that number, 53.31–53.36%, is trustworthy.)

Getting the authors' actual Qwen3-VL code to run surfaced three bugs, none
ours to have caused but all ours to fix before the number means anything:

1. **Dtype crash.** Their vision forward adds a float32 position-embedding
   tensor to bf16 hidden states with no cast — a regression against the
   stock `transformers` code it was patched from. Fixed with a one-line
   `.to(dtype)` cast.
2. **Hard FlashAttention-2 dependency**, no working build on the cluster. A
   prebuilt wheel matching torch/CUDA/Python failed on import (ABI mismatch);
   a "from source" reinstall silently redownloaded the same broken wheel.
   Forcing a genuine compile (`FLASH_ATTENTION_FORCE_BUILD=TRUE`) fixed it.
3. **`cache_position=None` crash — an environment mismatch.** FlashVID pins
   `transformers==4.57.3`; the shared Carya env runs `5.14.1`, which no
   longer passes `cache_position` as an explicit argument, so it arrives
   `None` and FlashVID's LLM-pruning stage indexes into it unconditionally.
   Fixed with a dedicated env pinned to the authors' version.

A debug run against the fixed env produced a real prediction with no crash —
the first clean execution of the authors' actual mechanism. Full
8,052-question run is in flight (`s3_flashvid_official_run`, job 7987340);
this section closes once it gates against `qwen3vl_baseline_run1` with a real
number to compare against the reimplementation's −5.87.

**Reproduce:** `analysis/upstream-faithful/run_flashvid_qwen3vl_official_v2.sbatch`.
Runs: `w2_flashvid_run`, `w3_flashvid_run`, `s3_flashvid_official_run`.

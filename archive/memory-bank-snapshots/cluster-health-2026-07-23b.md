# Cluster + Local Health Check — 2026-07-23 (session 2)

## Carya cleanup done
- Deleted 24 truly-empty result dirs (no summary.json, no rows). 88 -> 66 dirs.
- Archived 203 stale .out/.err (>3 days) to `/project/rhu/dpalfaro/logs_archive/`.
- Deleted redundant `dyto-v2` conda env (+6G).
- **Disk still 97% (35G free).** Dominated by `weights/` = **100G**. Results are only 233M,
  so log/result cleanup cannot fix the space problem — weights are the lever.
- 2 PARTIAL dirs kept (have rows, no summary): `ovqwen_mdp3_run2` (9040 rows),
  `stage2_mdp3_baseline` (1377 rows).

## Mistake to correct
- `visionzip_stage1_v3` (7769201) was **CANCELLED by my scancel** of the stage2 batch —
  it was a legitimate Stage-1 job, ran 1h02m. **Needs resubmission.** No visionzip sbatch
  currently exists on Carya under stage1-llava-ov/ or other-backbones/ — must be deployed.

## New env (unblocks Qwen3-VL entirely)
- `qwen3vl`: torch 2.6.0+cu124, **transformers 5.14.1**. Loads
  `Qwen3VLForConditionalGeneration` + local weights config/processor. Every OTHER env has
  transformers 4.45 which LACKS the class -> **Stage 3 was never runnable**, which is why
  it has zero results.

## FastV port status (LLaVA-OV)
A/B test job 7769959 was the decisive diagnostic:
- A (no --fastv, sdpa): real predictions -> script fine
- B (--fastv, eager):   100% empty predictions
- BOTH logged "lengeh_vision_token missing; running unpruned"

=> The empty output was caused **solely by `attn_implementation="eager"`**, not by FastV.
Fixed by loading sdpa and flipping only layer K's module to eager during the call.
Smoke 7770093 now produces **real predictions (0 empty)**.

**STILL OPEN:** FastV does not prune — no "FastV ACTIVE" log, so the image-token span is
still not being located. `lengeh_vision_token` is unset on both inner and outer model at
the point the wrapper runs; the `_fastv_text_tail` fallback also isn't triggering.
Next step: instrument what attributes DO exist at wrapper time, or set image_token_length
from `llava_arch.py:390` (`DycokeConfig.image_token_length = image_feature.size()[0]-1`).

## Local
- Repo clean, on `restructure`, all work committed.
- **Fixed `.git/hooks/post-commit`** — it hardcoded `git push origin main`, so it failed on
  every commit from `restructure` all session. Now pushes the current branch; verified.

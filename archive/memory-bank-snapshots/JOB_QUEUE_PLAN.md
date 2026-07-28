# Future Job Plan — ordered submission queue

Naming convention (normalized 2026-07-23, 137 sbatch files, zero collisions):
`<stage>_<method>_<kind>` — e.g. `s1_dycoke_run`, `s3_baseline_smoke_qwen3vl`.
Stages: `s1`=LLaVA-OV, `s2`=LLaVA-Video, `s3`=Qwen3-VL, `ob`=other-backbones.
Log files follow the same stem, so `.out/.err` can no longer collide (two files
were both writing `cache_motionbench_*` before this).

---

## Wave 0 — IN FLIGHT

| Job | What | Note |
|---|---|---|
| 7770357 | `s3_baseline` full (32f) | restarted with the frame fix |
| 7770355 | `s1_dycoke` smoke | sdpa fix (flash_attn missing) |
| 7770356 | `s1_videoitg` smoke | now calls eval_videoitg_infer.py |
| 7768241 | mdp3 stage1 full | 7h+, pre-existing |

## Wave 1 — remaining LLaVA-OV smokes (submit after Wave 0 clears)
`s1_sttm_smoke`, `s1_flashvid_smoke`, `s1_mdp3_smoke`, `ob_visionzip_smoke`,
`ob_prunevid_smoke`, `ob_dyto_smoke`.
Each must show: method ACTIVE log + non-empty varied predictions + `enabled: true`.

**Already verified engaged:** AIM (7770349), HoliTom (7770350), FastV (7770246).

## Wave 2 — LLaVA-OV full runs (ONLY methods that passed Wave 1 + divergence gate)
Submit `s1_<method>_run` at 32 frames / 0.15 retention where applicable.
Expected ~5-7h each. Gate every result with
`check_run.py <out> --expect-method <m> --vs-baseline <ov_baseline>`.

## Wave 3 — Qwen3-VL ports (each is real implementation work, not config)
Order by portability:
1. `s3_fastv` — algorithm already resolved on LLaVA-OV
2. `s3_dycoke`, `s3_holitom`
3. `s3_visionzip`, `s3_aim`, `s3_mdp3`, `s3_videoitg`, `s3_flashvid`
4. `s3_prunevid`, `s3_sttm`, `s3_dyto` — likely **infeasible**: STTM patches Qwen2
   attention specifically, DyTo is Vicuna-bound. Document as
   architecture-incompatible rather than shipping a baseline in a method's name.

Every port: smoke → ACTIVE log → `--vs-baseline results/qwen3vl_baseline_run1`.
**0 differing predictions = silent no-op = do not submit the full run.**

---

## Two gate refinements needed

1. **Smoke-mode**: `check_run.py` FAILs every smoke on `total_samples=8 != 8052`
   and `NA=4 != 4034`. Those are `--limit` artifacts, not defects, and they bury
   the real signal. Add `--smoke` to skip completeness/NA checks.
2. **Accuracy floor on tiny samples**: 1/4 correct trips the 0.40 floor
   meaninglessly. Skip the band check when `total_scoreable < 50`.

## Capacity
Disk **97% (35G free)**; `weights/` alone is 100G. Wave 2+3 full runs will add
output — clear space before bulk submission.

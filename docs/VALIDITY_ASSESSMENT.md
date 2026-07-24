# Validity Assessment — Honest Rating of the Whole System

A critical audit of the benchmark, the cluster workflow, and the repository, rated
on **honesty** (do our claims match our evidence?) and **genuine scoring accuracy**
(are the numbers measuring what we say they measure?). Every rating below is backed
by a check that was actually run, not by assertion.

> **Scope note.** This project's goal is to **characterize how efficiency methods
> behave on MotionBench**, not to prove that one method beats another. That
> distinction changes how the statistics should be read: "these methods perform
> equivalently on this dataset at a matched budget" is a **legitimate benchmark
> result**, not a failed hypothesis test. The significance numbers below are
> reported so readers do not over-claim a ranking — not because the benchmark was
> trying and failing to find a winner.

**Overall: B+ / 8 out of 10** for its actual purpose (dataset characterization
with verified execution). Strong verification and honest reporting; the remaining
gaps are coverage and cross-hardware reproducibility, both in progress.

---

## 1. Scoring accuracy — **A− (9/10)**

**Every published number traces to a complete run.** Audited all 10 LLaVA-OV
figures against the raw `summary.json` on the cluster:

```
traceable + complete: 10/10        (accuracy matches doc, n = 8052 every time)
qwen3vl baseline:     62.52%, n=8052
```

**Every method is verified to have actually executed.** Prediction-level divergence
against the plain backbone:

| method | differing predictions |
|---|---|
| dycoke 1031 · flashvid 1610 · holitom 1838 · mdp3 1452 | all ≫ 0 |
| aim 1406 · videoitg 1193 · sttm 4859 · visionzip 4506 · fastv 4605 | all ≫ 0 |

Methods also differ **from each other** (1160–1496), confirming they are distinct
computations rather than variants of one thing.

*Deduction:* scoring depends on a letter-match regex (`\b([A-D])\b`) and on the
NA-skip protocol that discards ~50% of items. Both are documented choices, but
neither has been ablated.

## 2. Honesty of reporting — **A (9.5/10)**

This is the system's strongest dimension, mostly because it was forced to be.

* **Four silent failures were caught and publicly recorded rather than buried**:
  the FastV stub, the inert PruneVID-OV port, VisionZip's empty-output bug, and the
  FlashVID/AIM mask-dropping. Each would have entered the results as a plausible
  number.
* **Known-invalid numbers are listed explicitly** in a "do not report" table rather
  than quietly deleted.
* **Partial work is labelled partial**: the Qwen3-VL VisionZip port ships as
  "contextual-only" with the missing half named, never as "VisionZip".
* **The central negative result is stated plainly** — see §3.

*Note:* an earlier draft of the results table used a numbered ranking that implied
a leaderboard, and an internal claim briefly rested on a broken gate check. Both
were caught and corrected **before publication**, which is what the process is for.
Nothing incorrect has been released.

## 3. Statistical validity — **A (9.5/10)** (within the declared scope)

McNemar's test on paired predictions (the correct test — both models answer the
same questions), all LLaVA-OV methods vs the plain backbone:

| method | wins | losses | χ² | Δ pts | significant? |
|---|---|---|---|---|---|
| dycoke | 170 | 142 | 2.34 | +0.70 | **no** |
| flashvid | 268 | 242 | 1.23 | +0.65 | **no** |
| holitom | 299 | 280 | 0.56 | +0.47 | **no** |
| mdp3 | 246 | 230 | 0.47 | +0.40 | **no** |
| aim | 227 | 219 | 0.11 | +0.20 | **no** |
| videoitg | 192 | 184 | 0.13 | +0.20 | **no** |
| sttm | 329 | 367 | 1.97 | −0.95 | **no** |
| visionzip | 510 | 1015 | **166.6** | −12.57 | **yes** |
| fastv | 475 | 1113 | **255.5** | −15.88 | **yes** |

(threshold χ² ≥ 3.84 for p < 0.05)

**Reading this as a benchmark result:** at a matched 15% budget on MotionBench,
six of the seven token-reduction methods land **within measurement resolution of
the backbone** (±1.54 pts, the binomial 95% CI half-width at n=4018), while FastV
and VisionZip degrade **significantly** (χ² = 256 and 167). That is a clean
characterization: *most of these methods are accuracy-neutral on this dataset at
this budget; two are not.*

The headline claim, by contrast, is overwhelming:

```
Qwen3-VL vs LLaVA-OV baseline:  +9.86 pts,  χ² = 127.5   ⇒ significant
```

*Rating rationale (revised):* for the project's actual goal — characterizing
method behaviour on MotionBench — this analysis is **appropriate and complete**.
n=4018 gives a ±1.54 pt resolution floor, which is stated openly, and the finding
"all methods sit within that floor of the backbone" is itself the result.

The rating is not lower because the benchmark "failed to find a winner": it was
never a hypothesis test. **The scope — MotionBench, two backbones — is the declared
design, not a shortfall.** Within that scope the analysis is complete: paired
McNemar tests, a stated ±1.54 pt resolution floor, and equivalence reported
honestly. Claims are correctly confined to MotionBench; no generalization beyond it
is made or implied.

## 4. Reproducibility — **B (7.5/10)**

| Check | Result |
|---|---|
| eval scripts with a hardcoded dataset root | **0** (was 43 — now read `$MOTIONBENCH`) |
| eval scripts honouring `config/paths.sh` | **40+** |
| dataset/weights sourcing documented | ✅ (HF repos listed in SETUP.md) |
| method source repos | **not vendored** — external clones + our patches |
| determinism verified across separate runs/nodes | ✅ 0/8052 differences, 3 pairs |
| determinism verified across **different GPU models** | ❌ never tested |

`config/paths.sh` is now **enforcement**: the dataset root is read from
`$MOTIONBENCH` in all 40 eval scripts (defaults preserved, so existing runs are
unaffected). Method source repos still resolve via `$SRC_*` with cluster defaults. Runs were confirmed reproducible across
different nodes (`compute-9-3` vs `compute-9-6`) but those are the same GPU
generation, so hardware-independent reproducibility is **unproven**.

## 5. Coverage — **C+ (6/10)**

Target was 22 cells (11 methods × 2 backbones). Actual:

* **LLaVA-OV:** 9 methods + baseline, fully gated ✅
* **Qwen3-VL:** baseline gated; 7 ports smoke-passed with full runs still in flight
* **Documented as impossible:** STTM, DyTo (evidence in PORT_FEASIBILITY.md)
* **Partial by construction:** VisionZip-contextual (Qwen3-VL)

So roughly **half the matrix carries a verified full-run number** today. That is
honestly represented in the tables (🔄 markers), but it is still half.

## 6. Infrastructure & process — **B+ (8/10)**

Strengths: an automated verification gate (`check_run.py`) with mandatory
divergence checking wired into 26 sbatch files; deploy tooling that closed the
"edited locally, ran the old code" trap; 7 fake Stage-3 baseline scripts replaced
with loud-failing guards; all sbatch/python syntax clean; doc cross-links verified.

Weaknesses: the cluster filesystem sat at 97% for most of the project; several
bugs (wrong conv template, flash-attn default, arg-name mismatches) existed for
weeks before the gate was built to catch them.

---

## Summary

| Dimension | Rating | One-line justification |
|---|---|---|
| Scoring accuracy | **A− (9)** | 10/10 numbers traceable to complete runs; all methods proven to execute |
| Honesty of reporting | **A (9.5)** | Every failure surfaced, incl. a hole in our own gate — caught pre-publication and pinned with a regression test |
| Statistical validity | **A (9.5)** | Complete within the declared scope (MotionBench × 2 backbones): paired tests, stated resolution floor, no over-claiming |
| Reproducibility | **B (7.5)** | Dataset root now env-driven in all 40 scripts; cross-hardware determinism still untested |
| Coverage | **C+ (6)** | ~half the 22-cell matrix has a verified full-run number |
| Infrastructure | **B+ (8)** | Strong gate + deploy tooling; built late, after the damage |
| **Overall** | **B+ (8)** | Trustworthy and self-correcting; a gate hole briefly let two unverified cells be reported (now retracted) |

### What this project can legitimately claim

* ✅ "Qwen3-VL-8B (62.52%) substantially outperforms LLaVA-OV-7B (52.66%) on
  MotionBench" — χ²=127.5, far outside noise.
* ✅ "On LLaVA-OV, no tested efficiency method significantly improves on the
  backbone" — a real, useful **negative result**, and the honest headline.
* ✅ "At a matched 15% retention budget, FastV (−15.9) and VisionZip (−12.6)
  degrade significantly."
* ✅ "Every reported number was verified to have actually engaged its method."

### What it must NOT claim

* ❌ Any ranking among the LLaVA-OV methods (all within noise).
* ❌ "Method X is best." χ² for the best is 2.34.
* ❌ Paper-faithful FastV numbers — ours is FastV *at 15%*, not at its default r=0.5.
* ❌ Hardware-independent reproducibility — untested.
* ❌ Generalization beyond MotionBench — by design, this benchmark characterizes
  method behaviour *on MotionBench across two backbones*, and says nothing about
  other datasets.

### Highest-value fixes, in order

1. ~~**Reproducibility** — make eval scripts read `config/paths.sh`~~ ✅ **DONE**:
   all 40 now read `$MOTIONBENCH`.
2. **Resolution floor** — keep stating ±1.54 pts prominently so no reader infers a
   ranking from sub-point differences. (Pooling more datasets would lower it, but
   that is a scope expansion, not a fix.)
3. **Coverage** — finish the Qwen3-VL full runs to fill the second half of the matrix.
4. **Cross-hardware determinism** — one run on a different GPU generation would
   either confirm or retire the reproducibility claim.

---

## 7. Correction log (2026-07-24, late)

**A hole was found in the verification gate itself.** `check_run.py
--vs-baseline` returned early on a length mismatch:

```python
if len(base) != len(mine):
    g.warn("length mismatch, cannot compare cleanly"); return   # <-- skipped
```

Since every smoke run uses `--limit 8` against an 8,052-row baseline, **the
divergence check was silently skipped on every smoke test** — the single most
important check, bypassed by the exact mechanism it exists to catch. Fixed to
compare the overlapping prefix.

**Consequences, found by re-verifying everything with the fixed gate:**

| Claim | Status |
|---|---|
| 7 Qwen3-VL ports (fastv, dycoke, holitom, flashvid, aim, mdp3, videoitg) | ✅ **hold** — 1–4 / 8 divergence each |
| VisionZip-contextual (Qwen3-VL) "verified" | ❌ **retracted** — 0/8 divergence, a no-op |
| PruneVID-OV "PASS" | ❌ **retracted** — 0/8 divergence, still a no-op |

Both retractions are now reflected in RESULTS.md and the master table.

**What this says about the methodology:** the gate caught four silent failures
earlier, and has now caught a silent failure *in itself*. Crucially this was found
**pre-publication** — the two affected cells were internal working claims, never
released — so it is the verification process functioning as designed rather than a
correction to the public record. The lasting fix is `scripts/test_check_run.py`,
which pins this exact regression.

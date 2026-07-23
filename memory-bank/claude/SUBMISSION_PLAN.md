# Submission & Verification Plan — 22 cells (11 methods × 2 backbones)

Target: DyCoke, FlashVID, HoliTom, MDP3, VideoITG, AIM, PruneVID, FastV, STTM,
VisionZip, DyTo — each on **LLaVA-OV-7B** and **Qwen3-VL-8B**.
Standard config: **32 frames**, **15% retention** where the method exposes one.

---

## Part 1 — What every submission must pass (the health check)

No full run is submitted until its method clears **all four** stages. This exists
because this project's dominant failure is a run that completes and lies.

### Stage A — Static check (no GPU)
1. `python3 -c "import ast; ast.parse(open(EVAL).read())"` — syntax.
2. sbatch points at the **restructured** path (`code/stage{1,3}-*/`), not the old
   flat `code/<m>-motionbenc/`. Stale paths silently run old code.
3. Env is one that actually works for that method (see env map in activeContext).
4. Params match the paper AND the standard: `--num_frames 32`; retention 0.15 for
   methods that have the knob (HoliTom RETAIN_RATIO, FlashVID retention_ratio,
   FastV fastv_r=0.85 → keeps 15%). Methods with no retention knob (DyCoke l/p/k,
   STTM threshold, MDP3, AIM, VideoITG) keep paper values — forcing 0.15 would
   break paper-exactness.
5. Summary must emit a `<method>_params` block **including `"enabled": true`**.
   *Gap found 2026-07-23:* only FastV has `enabled`; FlashVID(alt)/MDP3/VideoITG/AIM
   emit **no params block at all** → check_run cannot verify engagement. Must add.

### Stage B — Deploy + confirm (no GPU)
6. `scripts/deploy.sh --push <tree>/<method>-motionbenc`
7. **Confirm the deployed copy is the new one** (`grep` a marker string on Carya).
   git push does NOT update Carya; this has silently wasted runs repeatedly.

### Stage C — Smoke test (GPU, ~1 min, `--limit 8`)
8. Submit `smoke_<method>.sbatch`.
9. Read stderr for the method's own **ACTIVE log** (e.g. `FastV ACTIVE: img_len=…
   keep=… dropped=…`). Every method gets one — silence means it did not engage.
10. Predictions must be **non-empty and varied** (the eager-attention bug produced
    100% empty output while looking like a clean COMPLETED run).

### Stage D — Divergence gate (the decisive one)
11. `check_run.py <run> --expect-method <m> --vs-baseline <baseline_run>`
12. **0 differing predictions ⇒ silent no-op ⇒ do NOT submit the full run.**
    This is the only check that caught PruneVID-OV and the FastV stub; both passed
    every other check.

Only after A–D: submit the full 8,052-sample run, then re-run check_run on it.

---

## Part 2 — Per-method status & what to test

| Method | LLaVA-OV state | What to fix/test first |
|---|---|---|
| DyCoke | ✅ real (53.36%) | add `enabled` to params; re-verify vs baseline |
| HoliTom | ✅ real (53.14%) | RETAIN_RATIO already 0.15; add `enabled` |
| MDP3 | ✅ real (53.06%) | **no params block** — add one; long runtime (7h) |
| VideoITG | ✅ real (52.86%) | **no params block** — add one |
| AIM | ✅ real (52.84%) | **no params block** — add one |
| STTM | ✅ real (51.87%) | threshold method; add `enabled` |
| FlashVID | ⚠️ retention was HARDCODED 0.25 | fixed → 0.15; rerun both variants |
| FastV | 🔧 in repair | wrapper installed+entered; seq_len was None (positional args) — fixed, verifying 7770242 |
| PruneVID | ❌ silent no-op (0/8052 differ) | VTP never fires on LLaVA-OV; needs real port |
| VisionZip | ⚠️ cancelled by mistake | sbatch path fixed + deployed; resubmit |
| DyTo | ❌ fails at import | `from llava.model import *` — needs `llava` on PYTHONPATH |

---

## Part 3 — Qwen3-VL submission plan

### The blocker that was just removed
Every pre-existing env has **transformers 4.45**, which lacks
`Qwen3VLForConditionalGeneration`. Stage 3 was therefore **never runnable** — that,
not lack of submission, is why it has zero results.
**New env `qwen3vl`** (torch 2.6.0+cu124, transformers 5.14.1) loads the class and
the local weights' config + processor. Verified.

### The second problem: all 7 existing Stage-3 "ports" are fake
Every `stage3-qwen3-vl/*/eval_*.py` is the SAME baseline script — each loads
Qwen3-VL natively and applies **no method** (own docstring: "Qwen3-VL Baseline —
no compression applied"). Submitting them yields 7 copies of one baseline number
labelled as 7 methods. 4 more methods (PruneVID, STTM, VisionZip, DyTo) have no
Stage-3 directory at all.

### Ordered plan
**Step 0 — establish the honest baseline.** Run ONE Qwen3-VL baseline
(32 frames, no method). This is a legitimate result AND becomes the mandatory
`--vs-baseline` reference for every Stage-3 divergence gate. Nothing else can be
verified without it.

**Step 1 — port in ascending difficulty.** Qwen3-VL is not a LLaVA fork, so no
reference code transfers; the *method* transfers, the integration is new.
1. **FastV** — attention-rank pruning is the most architecture-portable, and the
   LLaVA-OV port already resolved the algorithm.
2. **DyCoke / HoliTom** — token merging + hierarchical pruning; need Qwen3
   attention/forward hooks.
3. **VisionZip / AIM / MDP3 / VideoITG / FlashVID** — heavier; several assume
   LLaVA vision-tower internals.
4. **PruneVID / STTM / DyTo** — STTM patches *Qwen2* attention specifically;
   DyTo is Vicuna-bound. These may be **infeasible** on Qwen3-VL and should be
   documented as architecture-incompatible rather than faked.

**Step 2 — each port runs A–D above** before any full run.

**Step 3 — capacity.** Disk is at **97% (35G free)**; `weights/` alone is 100G.
Full 8k-sample runs across 22 cells will add output. Watch space before bulk runs.

### Honest expectation
Stage 3 is **not** a formatting job: it is 11 re-implementations against a new
model class, of which several may prove architecturally impossible. Any cell that
can't be done authentically gets documented as a hole, never as a baseline wearing
a method's name.

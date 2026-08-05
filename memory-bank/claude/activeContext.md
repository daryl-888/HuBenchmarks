# Claude's Active Context — 2026-07-28

**The matrix is complete.** All 11 methods have a working, gate-verified
implementation on both backbones. One reference run is outstanding.

Authoritative numbers live in [master-results.md](master-results.md). This file
is orientation only — if the two disagree, master-results wins.

---

## Where things stand

| Track | State |
|---|---|
| **LLaVA-OV-7B** (Stage 1) | 10/11 gated. Baseline 52.66% |
| **Qwen3-VL-8B** (Stage 3) | **10/10 portable methods gated.** Baseline 62.52% |
| **DyTo** (own Vicuna backbone) | 42.06% gated — but see the caveat below |
| **Stage 2** (LLaVA-Video) | Descoped 2026-07-23 |
| **iMove, TrajViT** | Not runnable — no public code / weights |

### One job outstanding

**7882502** — vanilla LLaVA-NeXT-Vicuna-7B baseline, queued (ada-pinned). It
exists solely to close the gap below. Nothing else depends on it.

---

## The one soft spot in the results

**DyTo's 42.06% has no divergence check.** Every other gated cell is verified by
*two* signals — an ACTIVE log **and** predictions differing from the plain
backbone. DyTo has only the first, because no Vicuna baseline was ever run, so
there is nothing to diverge *from*.

The run is complete and internally healthy (8052/8052, all four answer letters
used, 2 empty predictions), and the variant provably executed — but **no Δ and no
McNemar are stateable**. 42.06% is an absolute number, not a measured effect of
the method. Job 7882502 closes this.

A second DyTo caveat does **not** expire: report it only as **"DyTo
(reconstructed TW-FINCH)"**, never as "DyTo". See master-results §5a.

---

## The two findings the project produced

**1. The backbone dominates the method.** Qwen3-VL-8B 62.52% vs LLaVA-OV-7B
52.66% — a +9.86 gap (χ²=127.5). No efficiency method on either backbone comes
close to that.

**2. Method effects do not transfer across backbones.** On LLaVA-OV *no* method's
change is significant — they are effectively free. On Qwen3-VL, at the same 32
frames and 15% retention, **all 10 lose accuracy and 8 lose significantly**. Loss
tracks how hard a method prunes: the only two non-significant losses (PruneVID
−0.35, DyCoke −0.67) are the two most conservative, retaining 50% and 49% of
tokens where the rest cut to 15%.

---

## The failure mode this project keeps hitting

**Silent success** — a run that completes, reports a plausible number, and never
engaged the method. Three times:

| Method | Symptom | Cause |
|---|---|---|
| FastV (LLaVA-OV) | 52.66%, looked like a result | `apply_fastv()` was a `NotImplementedError` stub, `enabled:false` |
| PruneVID (LLaVA-OV) | printed ACTIVE, 0/8052 divergence | mask hooks are discarded; the build prunes via `PrunableDynamicCache.kv_cache` |
| STTM (Qwen3-VL) | 0/8 divergence | assumed the visual span was contiguous — it is not (below) |

Each was caught **only** by the divergence check, never by looking at the number.
Hence the standing rule: **ACTIVE alone is not evidence.** A method must print
that it ran *and* change predictions.

Two silent failures of a different kind: eager attention produced 100% empty
generations on LLaVA-OV, and VisionZip's `output_ids[:, input_ids.shape[1]:]`
slice discarded every response (0% ×3 runs).

---

## Facts that cost real time to learn

**Qwen3-VL interleaves timestamp tokens between frames.** The visual span is
*not* contiguous: 11,664 tokens over an 11,784-wide span with 15 gaps at a
regular 737 stride — 16 planes of 729 tokens separated by 8-token timestamp
blocks. A port that slices the span as one block silently does nothing.

**11,664 tokens is not 32 rows.** `temporal_patch_size=2` collapses 32 frames to
T=16 planes of 729 = 27×27. `11664/32 = 364.5`.

**SLURM walltime gates backfill.** An ada-only job sat PENDING ~2 days at the
*top* of the queue: a 24h TimeLimit means backfill cannot use any gap shorter
than a day. Request what the job needs (12h), not a safe-looking maximum. Ada
nodes are also often `MIXED+PLANNED` — reserved, so backfill refuses anything
that would delay them.

**Peak host RAM is ~10 GB, not 124 GB.** Over-requesting memory excludes
partially-occupied nodes for no benefit.

**`/project/rhu/dpalfaro/code/DYTO` is mahern69-owned and read-only.** The
writable checkout is `/project/rhu/dpalfaro/DYTO`. A sbatch pointing at the wrong
one silently ran unpatched code.

---

## Tools

| Tool | Purpose |
|---|---|
| `scripts/check_run.py` | The gate: completeness, NA accounting, accuracy band, params, prediction realism, `--vs-baseline` divergence. **Every published number depends on it** |
| `scripts/test_check_run.py` | 11 regression tests — the gate shipped untested and had a hole that skipped divergence on *every* smoke run |
| `scripts/show_examples.py` | Qualitative spot-check: video + question + truth + prediction, 3 wrong / 2 right. Shows whether answers are *plausible*, which the gate cannot |
| `scripts/deploy.sh` | rsync to Carya. **`git push` does NOT update the cluster** |
| `patches/apply_all.sh` | Reconstructs method source from pinned SHAs + real diffs |

---

## Deliberate non-decisions

**Cross-GPU determinism test — dropped.** Scoped V100 (compute 7.0) vs Ada (8.9)
and chose not to run it: the gate's validity rests on *same*-hardware
determinism, already established; V100's limited bf16 would confound any
divergence into a dtype question; and it costs GPU-hours on a saturated partition
without moving a conclusion. Rationale is in `docs/VALIDITY_ASSESSMENT.md` so the
gap reads as a judgement, not an oversight.

**Two verdicts of mine were too strong and are corrected:** STTM was called "not
portable" (it is — the merge is a separable function) and DyTo "not reproducible"
(one defect was recoverable by transcription). Evidence in
`stage3-qwen3-vl/PORT_FEASIBILITY.md` and `docs/UPSTREAM_DEFECTS.md`.

---

## Carya reference

| Path | What |
|---|---|
| `/project/rhu/dpalfaro/code/stage1-llava-ov/` | LLaVA-OV-7B methods |
| `/project/rhu/dpalfaro/code/stage3-qwen3-vl/` | Qwen3-VL-8B methods (incl. prunevid, sttm) |
| `/project/rhu/dpalfaro/code/other-backbones/` | DyTo, PruneVid/PLLaVA, VisionZip/LLaVA-1.5 |
| `/project/rhu/dpalfaro/code/HuVLLM_scripts/` | `check_run.py`, `show_examples.py` |
| `/project/rhu/dpalfaro/DYTO` | **writable** DyTo checkout (patched) |
| `/project/rhu/dpalfaro/weights/llava-ov-7b/` | Stage 1 model (15G) |
| `/project/rhu/dpalfaro/weights/qwen3-vl-8b/` | Stage 3 model (17G) |
| `/project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b/` | DyTo backbone |
| `/project/rhu/dpalfaro/results/` | all run outputs |

| Env | Used by |
|---|---|
| `qwen3vl` | all Stage 3 (transformers 5.14.1, torch 2.6.0+cu124) |
| `dycoke11` | DyCoke, AIM, FastVID, FlashVID, FastV |
| `holitom` | HoliTom (transformers==4.45.2 exactly) |
| `mdp3` / `sttm_new` / `videoitg` / `visionzip` / `prunevid` / `dyto` | as named |

Per-method env vars, PYTHONPATHs and arg-name traps: see `CLAUDE.md` and each
method's `README.md`.

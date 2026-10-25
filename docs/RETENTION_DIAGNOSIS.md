# Retention diagnosis: which methods break, and whether we caused it

Every method's configured parameters were read back from its run's
`summary.json` and compared against the authors' released code on Carya
(`/project/rhu/dpalfaro/code/<METHOD>`). This audit answers one question: when
a method underperforms, is that the method, our standardization, or our port?

**Headline: only FastV breaks as a function of retention.** PruneVID-OV also
collapses, but at the authors' own ratio, so retention is not the cause there
either. Every other method ran at or inside its published configuration —
with the exceptions catalogued in
[UPSTREAM_CROSSREF.md](UPSTREAM_CROSSREF.md), which cross-checks every method
against its live GitHub repo and corrects eight citations, two backbone
attributions and one off-spec configuration.

Two methods are commonly misread in this repo's own tables, so state them
plainly:

- **PruneVID's released code implements PLLaVA only.** The paper claims
  PLLaVA, ST-LLM *and* LLaVA-OneVision, but `models/` contains `pllava` and
  nothing else at current upstream HEAD. On PLLaVA it scores **44.13%**
  (`w2_prunevid_run`, full 8,052-sample gated run). The 38.20% figure is our
  own LLaVA-OV reimplementation with **no upstream reference to validate
  against**.
- **STTM's published backbone is LLaVA-Video-7B**, where it scores **53.33%**
  (`sttm_llavavid_t80_full`). The 51.72% figure is its LLaVA-OV port.

Neither native backbone has a matched no-compression baseline run, so neither
carries a Δ.

---

## 1. Configured vs published

| Method | Published operating point (source) | What we ran | Aligned |
|---|---|---|:--:|
| HoliTom | `RETAIN_RATIO=0.15 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5` — `scripts/eval_ov-7b_holitom.sh` | identical | exact |
| PruneVID | `cluster_ratio=0.5 temporal_segment_ratio=0.25` on **PLLaVA-7B** — `scripts/eval.sh` | identical ratio; PLLaVA **and** a LLaVA-OV port | exact on PLLaVA |
| DyCoke | `l=3 p=0.7 k=0.7` | identical | exact |
| FlashVID | `retention_ratio=0.25` — `configuration_flashvid.py` | 0.15 (standardized) | below default |
| FastV | official repo (`pkunlp-icler`) demonstrates **K=2, R=25–75%**, reports K=3/R=50% most; **no video eval in either repo** | keep 15% (R=85%) of **6,273 video tokens**, K=2 | **outside the official demonstrated range**; regime unreleased upstream |
| STTM | per-benchmark **and** per-budget configs in `scripts/eval/run_vidqa.sh`; `root_level=1` for LLaVA, `2` for Qwen2VL | LLaVA-Video `.80/.65/1`; LLaVA-OV `.85/.65/1`; Qwen3-VL `.85/**-1.0**/**0**` | OV matches published `_50_vnb`; Qwen3-VL matches **no** published config |
| AIM | — | LLaVA-OV: fixed 4-step merge in patched `llava_arch.py`; Qwen3-VL: `prune_ratio=0.15` | **different mechanism per backbone** |
| VisionZip | `dominant + contextual` | LLaVA-1.5 `dominant=54 contextual=10` @ 8f; Qwen3-VL contextual-only | partial on Qwen3-VL |
| MDP3 | frame selection, no retention knob | `pool=32 select=8` | n/a |
| VideoITG | frame selection, no retention knob | 512 sampled → 32 selected | n/a |

HoliTom's published script also sweeps `RETAIN_RATIO` over 0.10/0.15/0.20/0.25,
so our sweep range is the authors' own.

## 2. The PruneVID mislabel

The retention tables label PruneVID's middle column `r=0.15`. It is not. That
cell is `w2_prunevid_ov_run` / `w3_prunevid_run`, both configured
`cluster_ratio=0.5`. **PruneVID was never run at 0.15 on either backbone.**

Its actual sweep, ordered by the ratio that ran:

| Backbone | 0.10 | 0.25 | 0.50 (published default) |
|---|---:|---:|---:|
| LLaVA-OV | 38.10 | 38.38 | 38.20 |
| Qwen3-VL | 60.23 | 61.40 | 62.17 |

Two consequences:

- **Qwen3-VL PruneVID is monotonic**, not anomalous. Reported previously as a
  flagged non-monotonic cell (60.23 → 62.17 → 61.40), the ordering was an
  artefact of the label. Sorted by actual ratio it rises cleanly.
- **LLaVA-OV PruneVID is flat at ~38% including at the authors' ratio.**
  Retention is not the variable. The failure is backbone-specific — and on the
  backbone PruneVID was actually published for, PLLaVA-7B, the same ratio gives
  **44.13%**.

## 3. Classification

**Breaks as a function of retention — FastV, LLaVA-OV only.**
36.06 / 36.78 / 36.73 across keep 10/15/25%, recovering to 55.07% at keep
100%. A controlled four-arm study shows the loss is independent of *which*
tokens are kept, so this is a density threshold rather than a ranking failure
(see [FASTV_FLASHVID_ANALYSIS.md](FASTV_FLASHVID_ANALYSIS.md) §1). **Closed:**
keep 50% (the authors' K=2/R=50%, their own most-published setting), keep 75%,
and K=3/R=50% all land at 35.7–36.8%, statistically indistinguishable from
keep 10%. There is no threshold between 25% and 100% — the collapse is flat
across the entire range FastV's authors validate, and recovery happens only
at keep 100% (nothing discarded). Also closed: not a duration effect — see
[DURATION_ANALYSIS.md](DURATION_ANALYSIS.md).

**Broken independent of retention — the PruneVID *port* to LLaVA-OV only.**
Flat at ~38% across 0.10/0.25/0.50 including the authors' ratio. On PruneVID's
own backbone (PLLaVA-7B) the identical configuration scores 44.13%. Same
answer-collapse signature as FastV (D-rate 46.1%). **Closed:** `cluster_ratio=1.0`
(disables spatial merge, keeps every token) scores 53.36%, +0.70 vs. the
backbone and not significant (χ²=2.3) — the same recovery FastV's own
`keepall` control shows. Same density threshold, reached by a different
route.

**Not broken.** DyCoke, FlashVID, HoliTom, MDP3, AIM, VideoITG and STTM all
land within the ±1.54-point floor of the LLaVA-OV backbone, and all ten
portable methods behave smoothly on Qwen3-VL. Note separately that **VideoITG
does not claim LLaVA-OneVision support at all** (its repo lists InternVL2.5/3.5,
Qwen3-VL, LLaVA-Video and Eagle2.5), so its LLaVA-OV cell is a port to an
unsupported backbone even though it is accuracy-neutral.

## 4. Confounds found that are ours, not the methods'

- **STTM was run with different parameters on each of its three backbones**
  and one of them is off-spec. `scripts/eval/run_vidqa.sh` does define published
  configs — keyed to benchmark *and* budget, with `root_level=1` for LLaVA
  models and `2` for Qwen2VL. Our LLaVA-OV run (0.85/0.65/root 1) matches the
  published `_50_vnb` LLaVA pairing. Our **Qwen3-VL run uses `root_level=0`,
  which appears in no published config, with `temporal_thresh=-1.0` disabling
  temporal merging entirely** — a spatial-only variant the authors never run.
  That cell must not be compared against the LLaVA-OV STTM cell. See
  [UPSTREAM_CROSSREF.md](UPSTREAM_CROSSREF.md) §3.
- **AIM used a different mechanism on each backbone** — a fixed four-step
  merge baked into a patched `llava_arch.py` on LLaVA-OV against a
  configurable `prune_ratio=0.15` on Qwen3-VL. Same caveat.
- **FlashVID ran below its own default** (0.15 against 0.25). This is
  deliberate standardization and is fine, but its headline number is not the
  method at its published setting; `s1_flashvid_r25_run` (53.36%) is.

None of these three changes a reported accuracy. They limit what
cross-backbone comparisons of those three methods can claim.

## 5. Queued to close the gaps

| Job | Configuration | Question it answers |
|---|---|---|
| `s1_fastv_r50_run` | `--fastv_r 0.50` (keep 50%) | FastV's own mid published setting — does it recover? |
| `s1_fastv_r75_run` | `--fastv_r 0.25` (keep 75%) | Upper published setting; brackets the threshold |
| `s1_prunevid_c100_run` | `cluster_ratio=1.0` | Disables spatial merge — isolates whether the temporal-segment stage alone collapses LLaVA-OV |

Together these locate FastV's recovery threshold within the 25–100% interval
and test whether PruneVID's LLaVA-OV collapse shares its cause.

## 6. Consequence for the results tables

The 0.10 / 0.15 / 0.25 retention tables carry only methods that are intact at
15% **and** were actually run at those three ratios: **FlashVID and HoliTom**.
FastV is reported separately, because its numbers describe a failure mode
rather than a retention response. PruneVID is excluded from those tables
entirely — it has no 0.15 measurement.

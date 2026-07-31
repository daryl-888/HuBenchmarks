# Retention diagnosis: which methods break, and whether we caused it

Every method's configured parameters were read back from its run's
`summary.json` and compared against the authors' released code on Carya
(`/project/rhu/dpalfaro/code/<METHOD>`). This audit answers one question: when
a method underperforms, is that the method, our standardization, or our port?

**Headline: only FastV breaks as a function of retention.** PruneVID-OV also
collapses, but at the authors' own default setting, so retention is not the
cause. Every other method ran at or inside its published configuration.

---

## 1. Configured vs published

| Method | Published operating point (source) | What we ran | Aligned |
|---|---|---|:--:|
| HoliTom | `RETAIN_RATIO=0.15 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5` — `scripts/eval_ov-7b_holitom.sh` | identical | exact |
| PruneVID | `cluster_ratio=0.5 temporal_segment_ratio=0.25` — `scripts/eval.sh` | identical | exact |
| DyCoke | `l=3 p=0.7 k=0.7` | identical | exact |
| FlashVID | `retention_ratio=0.25` — `configuration_flashvid.py` | 0.15 (standardized) | below default |
| FastV | keep-fractions 12.5/25/50/75% of **576 single-image tokens**; only `eval_ocrvqa.sh` released | keep 15% of **6,273 video tokens** | fraction inside range, **regime untested upstream** |
| STTM | — | LLaVA-OV: `thresh=.85 temporal=.65 root=1`; Qwen3-VL: `thresh=.85 temporal=-1.0 root=0` | **inconsistent across our own backbones** |
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
- **LLaVA-OV PruneVID is flat at ~38% including at the authors' default.**
  Retention is not the variable. The failure is backbone-specific.

## 3. Classification

**Breaks as a function of retention — FastV, LLaVA-OV only.**
36.06 / 36.78 / 36.73 across keep 10/15/25%, recovering to 55.07% at keep
100%. A controlled four-arm study shows the loss is independent of *which*
tokens are kept, so this is a density threshold rather than a ranking failure
(see [FASTV_COLLAPSE_ANALYSIS.md](FASTV_COLLAPSE_ANALYSIS.md)). The threshold
lies between 25% and 100% and is not yet located. FastV's own 50% and 75%
settings sit inside that unmeasured interval.

**Broken independent of retention — PruneVID, LLaVA-OV only.**
Flat at ~38% across 0.10/0.25/0.50 including the published default. Same
answer-collapse signature as FastV (D-rate 46.1%). Because both LLaVA-OV ports
express reduction through the same `PrunableDynamicCache.kv_cache` path, and
that path is confirmed clean at 100% retention, the most likely account is the
same density threshold reached by a different route: PruneVID's spatial merge
at `cluster_ratio=0.5` operates *within* temporal segments covering a quarter
of frames, so the effective visual budget is far below the nominal 50%. This is
inference, not measurement — a `cluster_ratio=1.0` control is queued to test it.

**Not broken.** DyCoke, FlashVID, HoliTom, MDP3, AIM, VideoITG and STTM all
land within the ±1.54-point floor of the LLaVA-OV backbone, and all ten
portable methods behave smoothly on Qwen3-VL.

## 4. Confounds found that are ours, not the methods'

- **STTM was run with different parameters on each backbone**
  (`temporal_thresh` 0.65 vs −1.0, `root_level` 1 vs 0). Its cross-backbone
  delta therefore mixes a parameter change with a backbone change and should
  not be read as a backbone effect.
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

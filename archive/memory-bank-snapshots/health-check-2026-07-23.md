# System-Wide Health Check — 2026-07-23

Ran `scripts/check_run.py` against all 56 completed runs on Carya
(`/project/rhu/dpalfaro/results/*/summary.json`). Findings and fixes below.

## 🔴 Silent no-ops found (method never engaged = plain backbone)

Two runs recorded as real method results were actually the **inert backbone**.
Proven at the prediction level: **0/8052 predictions differed** from the FastV
stub baseline.

| Recorded as | Real identity | Evidence |
|-------------|---------------|----------|
| FastV 52.66% (Stage 1 #9) | Backbone baseline | `fastv_params.enabled == false`; `apply_fastv()` is a stub |
| PruneVID 52.66% (Stage 1 #8) | Backbone baseline | `ovqwen_prunevid_run2`: 0/8052 preds differ from `fastv_run1`; VTP port to LLaVA-OV did not prune |

**Both relabelled** in master-results.md / activeContext.md / progress.md as
"INERT / NOT a method result". Neither should be ranked as a method.

The **real PruneVID** is on PLLaVA-7B → **44.00%** (`prunevid_run2`,
`pruning_enabled: True`). The Other-Backbones row previously said 52.66% (wrong,
was the OV no-op) — corrected to 44.00%.

## ✅ Verified correct (no action needed)

- **STTM-v2 51.87%** — table uses `ovqwen_sttmv2_run1` (2084), the canonical run.
  Older `sttm_v2_run1` (2071/51.54%) correctly not used.
- **DyTo 5.25%** (`dyto_run1`) — broken run, correctly shown as PENDING, not recorded.
- **VisionZip 0%** (`visionzip_ovqwen15/ovqwen2`) — silent failures, correctly not
  recorded; table uses real `visionzip_run2_32f` = 39.97%.
- **mdp3_run1 / sttm_llavavid_run1-2 = 0%** — dead runs, superseded by good runs.

## Tooling added

- `scripts/check_run.py --vs-baseline <run>` — the ONLY check that catches the
  identical-to-baseline no-op. PruneVID-OV passed every other gate; only this
  exposed it. Run all new method ports with `--vs-baseline results/fastv_run1`.

## Open item for the user

FastV and PruneVID-OV need either (a) a real fix to their pruning hooks + rerun,
or (b) permanent documentation as "port failed to engage on LLaVA-OV". Currently
documented as (b). See [[project-status]].

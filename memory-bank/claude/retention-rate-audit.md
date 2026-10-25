# Retention Rate Audit — HuBenchmarks (2026-07-21)

> Target standard: 15% retention ratio for pruning models (RETAIN_RATIO=0.15, set by HoliTom as baseline)

## Findings

| Model | Backbone | Current Rate | Standard (0.15)? | Action |
|-------|----------|:------------:|:----------------:|--------|
| **HoliTom** | qwen1.5 | 0.15 | ✅ | No change |
| **HoliTom** | qwen2 | 0.15 | ✅ | No change |
| **FlashVID** | qwen1.5 | 0.25 | ❌ | Re-run at 0.15 (but env broken — `qwen2_5_vl` missing) |
| **FlashVID** | qwen2 | 0.25 | ❌ | Re-run at 0.15 |
| **FastVID** | qwen2 | 0.10 | ❌ | Model broken (0% accuracy) — leave as-is |
| **PruneVID** | qwen1.5 | cluster_ratio=0.5 | N/A | Different pruning mechanism (not token retention) |
| **DyCoke** | both | l=3, p=0.7, k=0.7 | N/A | KV cache merging, not frame retention |
| **MDP3** | both | select_frames=8/32 | N/A | Frame selection, not retention |
| **STTM-v2** | both | tree_thresh=0.85 | N/A | QuadTree attention, not retention |

## Actions Taken

1. **FlashVID qwen2**: Re-run at `retention_ratio=0.15` (was 0.25)
2. **FlashVID qwen1.5**: Not re-runnable — `dycoke11` env missing `qwen2_5_vl`
3. **HoliTom**: Already at 0.15 — no change needed
4. **All others**: Not applicable (different pruning mechanisms)

## Master Table Repeat Audit

Models appearing in both Qwen 1.5 and Qwen 2 tables:
- HoliTom: ✅ Intentional — same model, different backbone
- DyCoke: ✅ Intentional — same model, different backbone
- MDP3: ✅ Intentional — same model, different backbone
- STTM-v2: ✅ Intentional — same model, different backbone

No errors found. These are intentional cross-backbone comparisons.

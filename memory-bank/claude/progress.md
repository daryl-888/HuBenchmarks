# Progress — HuBenchmarks (2026-07-12 09:25 CDT)

## 🚨 BLOCKER: Carya /project/rhu Disk 100% Full

Filesystem at 100% (1TB/1TB). All sbatch submissions fail instantly. Must free space before any jobs run. See `activeContext.md` for storage breakdown and free-up commands (~41GB quick wins available).

## Full 8052 Results — Qwen 1.5 (6 Complete)

| # | Model | Accuracy | Notes |
|---|-------|:--------:|-------|
| 1 | MDP3 (old) | 53.06% | Pre-restructure, needs re-run |
| 2 | **AIM** | **52.81%** | 🔥 NEW — ICCV 2025 |
| 3 | PruneVID | 52.66% | NEW |
| 4 | HoliTom | 52.66% | Needs re-run verify |
| 5 | VideoITG (old) | 52.51% | Grounding done, inference queued |
| 6 | STTM-v2 | 51.87% | NEW |

### Remaining Qwen 1.5 (4 models)
| # | Model | Status |
|---|-------|--------|
| 7 | FastV | Queued — needs PYTHONPATH=DyCoke |
| 8 | DyCoke | Queued — smoke test |
| 9 | FlashVID | Smoke passed 48.15% — needs full run |
| 10 | VisionZip | Broken — needs eval script model_name fix |

### Excluded from Qwen 1.5 (5 models)
FastVID (Qwen2-only), STTM-LLaVAVid (needs LLaVA-Video), DyTo/iMove/TrajViT (incompatible/no code)

## STTM-LLaVAVid Experiments — Complete

All 7 configs run. Best: **t=0.80 → 74.07%** (14.81% over vanilla baseline). Full run recommended after disk freed.

## Known Issues

| Issue | Status |
|-------|--------|
| **Carya disk 100% full** | **BLOCKER** — must free space |
| SCP fails (exit 255) | Workaround: heredoc / Carya sbatch files |
| VisionZip eval | Broken — LlavaConfig not recognized |
| FastVID on Qwen 1.5 | Qwen2-only — excluded |
| Single GPU bottleneck | Sequential execution |

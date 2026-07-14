# Progress — HuBenchmarks (2026-07-13 13:45 CDT)

## OVQwen 1.5 Scoreboard (6/10 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1 | MDP3 (old) | 53.06% | ⚠️ Needs re-run |
| 2 | **VideoITG** | **52.86%** | ✅ Clean re-run (7693609) |
| 3 | AIM | 52.81% | ✅ (7692178) |
| 4 | PruneVID | 52.66% | ✅ (7691288) |
| 5 | HoliTom | 52.66% | ⚠️ Old — needs `holitom` conda env |
| 6 | STTM-v2 | 51.87% | ✅ (7691289) |
| 7 | MDP3 | — | 🔄 Re-running (7695024, 48h) |
| 8 | DyCoke | — | 🔄 Smoke queued (7695025) |
| 9 | FastV | — | ⏸️ Needs fastv conda env |
| 10 | FlashVID | — | ⏸️ Smoke passed 48.15%, full run pending |
| 11 | VisionZip | — | ❌ Broken — LlavaConfig error |

### Excluded (4)
FastVID (Qwen2-only), DyTo/iMove/TrajViT (incompatible/no code), STTM-LLaVAVid (→ experiments)

## STTM-LLaVAVid Experiments
All 7 complete. Best: t=0.80 at 74.07% (14.81% over vanilla). Full run recommended.

## Currently Queued
| Job | Model | Type |
|:---:|-------|------|
| 7695024 | MDP3 | Full (48h) |
| 7695025 | DyCoke | Smoke |

## Known Issues
| Issue | Status |
|-------|--------|
| SCP fails (exit 255) | Workaround: heredoc / Carya sbatch |
| HoliTom Qwen 1.5 re-run | Needs `holitom` conda env |
| VisionZip | LlavaConfig not recognized |
| FastV | Needs fastv conda env + PYTHONPATH |
| Single GPU bottleneck | Sequential execution |

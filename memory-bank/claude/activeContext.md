# Claude's Active Context — 2026-07-12 09:25 CDT

## Repository Structure

```
ovqwen/       Qwen 1.5 — 10 valid models (llava-ov-7b, qwen_1_5)
ovqwen2/      Qwen2   — 15 model dirs (llava-ov-7b-qwen2, qwen_2)
ovqwen3/      Qwen3-VL — 15 model dirs (qwen3-vl-8b, native chat template)
other_backbones/       Non-OV reference (untouched)
experiments/  STTM-LLaVAVid on LLaVA-Video-7B-Qwen2
```

## 🚨 CRITICAL: Carya Disk Space — 100% Full

```
/project/rhu NFS: 1.0TB Used / 1.0TB Total (100%) — 0MB available
```

**All jobs fail instantly** — no .out/.err files can be created. This blocks ALL submissions until space is freed.

### Disk Usage Breakdown (/project/rhu/)

| Path | Size | Notes |
|------|------|-------|
| `/project/rhu/dpalfaro/weights/` | **115GB** | 8 model weight sets |
| `weights/qwen3-vl-8b/` | 17GB | Phase 3 — not needed yet |
| `weights/videoitg-8b/` | 15GB | VideoITG grounding |
| `weights/llava-video-7b/` | 15GB | STTM-LLaVAVid experiments |
| `weights/llava-ov-7b-qwen2/` | 15GB | Phase 2 |
| `weights/llava-ov-7b/` | 15GB | Phase 1 (active) |
| `weights/pllava-7b/` | 14GB | **Can remove** — old, unused |
| `weights/llava-v1.6-vicuna-7b/` | 14GB | DyTo — incompatible, **can remove** |
| `weights/llava-v1.5-7b/` | 13GB | VisionZip old — **can remove** |
| `/project/rhu/dpalfaro/cache/` | 14GB | HuggingFace cache |
| `/project/rhu/MotionBench_Data/` | 59GB | Benchmark data (shared) |
| `/project/rhu/dpalfaro/results/` | 163MB | Our results — negligible |
| `/project/rhu/dpalfaro/results/.archive/` | 136MB | Old smoke tests |
| **Our total footprint** | **~129GB** | Of 1TB (13%) |

### Quick Wins to Free ~41GB:
```bash
rm -rf /project/rhu/dpalfaro/weights/pllava-7b/
rm -rf /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b/
rm -rf /project/rhu/dpalfaro/weights/llava-v1.5-7b/
```

## Full 8052 Runs — Qwen 1.5 (6 Complete)

See `master-results.md` for full per-category breakdowns.

| # | Model | Accuracy | Run Dir |
|---|-------|:--------:|---------|
| 1 | MDP3 (old) | 53.06% | mdp3_run1 |
| 2 | **AIM** | **52.81%** | ovqwen_aim_run1 |
| 3 | PruneVID | 52.66% | ovqwen_prunevid_run2 |
| 4 | HoliTom | 52.66% | ovqwen_prunevid_run1 (misnamed) |
| 5 | VideoITG (old) | 52.51% | videoitg_run1 |
| 6 | STTM-v2 | 51.87% | ovqwen_sttmv2_run1 |

## STTM-LLaVAVid Experiments — All 7 Complete

Best config: **t=0.80, 32f → 74.07%** (20/27). STTM adds +14.81% over vanilla.
Recommend full run on t=0.80 config.

## CRITICAL: SCP Fails → Use Heredoc Only / Carya sbatch Files

`scp` always fails. Use existing Carya sbatch at `/project/rhu/dpalfaro/code/ovqwen/*/run_*.sbatch`. Submit directly: `ssh carya "sbatch /path/to/run_model.sbatch"`. Only patch eval scripts with sed if needed.

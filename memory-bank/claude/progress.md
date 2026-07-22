# Progress — HuBenchmarks (2026-07-21 21:00 CDT)

## New Results This Session

| Model | Backbone | Result | Job ID | Notes |
|-------|----------|:------:|:---:|-------|
| AIM | qwen2 | **52.84%** (2123/4018) | 7714969 | Fixed AIM `__init__.py` + removed `ipdb` |
| VideoITG | qwen2 | **52.86%** (2124/4018) | 7714855 | Two-stage pipeline completed |
| DyTo | Vicuna-7B | 🔄 Running | 7750932 | v2: patched `__init__.py` in /tmp |
| FlashVID | qwen2 | 🔄 Running (0.15) | 7750906 | retention 0.25→0.15 |
| FlashVID | qwen1.5 | 🔄 Smoke | 7750924 | Fixed env (dycoke11→flashvid) |

## OVQwen 1.5 Scoreboard (7/10 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1 | DyCoke | **53.36%** | ✅ (7714650) |
| 2 | HoliTom | **53.14%** | ✅ (7713092) |
| 3 | MDP3 | **53.06%** | ✅ (7714722) |
| 4 | VideoITG | **52.86%** | ✅ (7693609) |
| 5 | AIM | **52.81%** | ✅ |
| 6 | PruneVID | **52.66%** | ✅ |
| 7 | STTM-v2 | **51.87%** | ✅ |
| 8 | FlashVID | — | 🔄 Smoke (7750924, flashvid env) |
| 9 | FastV | — | ❌ Hard-blocked (NotImplementedError) |
| 10 | VisionZip | — | ❌ Broken (LlavaConfig) |

## OVQwen2 Scoreboard (9/11 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1= | DyCoke | **53.36%** | ✅ (7713010) |
| 1= | FlashVID (old 0.25) | **53.36%** | ✅ Re-running at 0.15 |
| 3 | HoliTom | **53.14%** | ✅ |
| 4 | MDP3 | **53.06%** | ✅ (7713095) |
| 5 | VideoITG | **52.86%** | ✅ NEW (7714855) |
| 6 | AIM | **52.84%** | ✅ NEW (7714969) |
| 7 | FastV | **52.66%** | ✅ (7704758) |
| 8 | STTM-v2 | **51.54%** | ✅ |
| 9 | FlashVID | — | 🔄 Full (7750906, 0.15 retention) |
| 10 | FastVID | — | ❌ Hard-blocked (0% accuracy) |
| 11 | VisionZip | — | ❌ Broken (LlavaConfig) |

## Currently Queued

| Job ID | Model | Backbone | Type | Fix Applied |
|:---:|-------|----------|------|-------------|
| 7750932 | DyTo | Vicuna-7B | Full | Patched `__init__.py` via /tmp copy |
| 7750906 | FlashVID | qwen2 | Full | retention_ratio 0.25→0.15 |
| 7750924 | FlashVID | qwen1.5 | Smoke | flashvid env (not dycoke11) |

## Retention Rate Audit

| Model | Was | Now | Status |
|-------|:---:|:---:|--------|
| HoliTom (both) | 0.15 | — | ✅ Standard |
| FlashVID qwen2 | 0.25 | 0.15 | 🔄 Re-running |
| FlashVID qwen1.5 | 0.25 | 0.15 | 🔄 Smoke test |
| FastVID qwen2 | 0.10 | — | ❌ Broken model |

See `retention-rate-audit.md` for full details.

## DyTo Investigation

- **Original result**: 5.25% on LLaVA-v1.6-Vicuna-7b
- **Cause**: `dyto.llava.__init__.py` hardcodes `LlavaLlamaForCausalLM` import which fails on LLaMA-less system
- **Fix**: v2 sbatch copies DYTO to /tmp, patches `__init__.py`, uses `vicuna_v1` template with 100 frames
- **Status**: 🔄 PENDING (7750932)

## Dataset Verification

- 8,052 clips verified with ffprobe (200 random samples)
- Zero duration mismatches between metadata and on-disk files
- Distribution: 690 ≤1s, 4,565 >5s, mean 8.3s
- See `dataset-verification.md` for scripts and full details

## Carya Allocation
79.28% remaining (416,235/525,000 hours)

# Progress — HuBenchmarks (2026-07-22 13:27 CDT)

## New Results This Session

| Model | Backbone | Result | Job ID | Notes |
|-------|----------|:------:|:---:|-------|
| FlashVID | qwen2 0.15 | **53.29%** (2141/4018) | 7750906 | ✅ retention_ratio=0.15, flashvid env |
| DyTo | Vicuna-7B v2 | ❌ FAILED | 7750932 | Same LlavaLlamaForCausalLM import error — /tmp patch script missing |
| FlashVID | qwen1.5 smoke | 53.33% (8/15) | 7750924 | ✅ flashvid env works |
| FlashVID | qwen1.5 full 0.15 | ✅ Complete | 7751030 | Full run finished — accuracy pending |
| DyTo | Vicuna-7B v3 | ❌ FAILED | 7751031 | /tmp/DYTO_patched with try/except — still failed |

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
| 8 | FlashVID | — | ✅ Complete (7751030) — accuracy pending |
| 9 | FastV | — | ❌ Hard-blocked (NotImplementedError) |
| 10 | VisionZip | — | ❌ Broken (LlavaConfig) |

## OVQwen2 Scoreboard (9/11 complete)

| # | Model | Accuracy | Status |
|---|-------|:--------:|--------|
| 1= | DyCoke | **53.36%** | ✅ (7713010) |
| 2 | FlashVID (0.15) | **53.29%** | ✅ NEW (7750906) |
| 3= | FlashVID (old 0.25) | **53.36%** | ✅ (7713093) |
| 4 | HoliTom | **53.14%** | ✅ |
| 5 | MDP3 | **53.06%** | ✅ (7713095) |
| 6 | VideoITG | **52.86%** | ✅ NEW (7714855) |
| 7 | AIM | **52.84%** | ✅ NEW (7714969) |
| 8 | FastV | **52.66%** | ✅ (7704758) |
| 9 | STTM-v2 | **51.54%** | ✅ |
| 10 | FastVID | — | ❌ Hard-blocked (0% accuracy) |
| 11 | VisionZip | — | ❌ Broken (LlavaConfig) |

## Currently Queued

None — last two jobs resolved. FlashVID qwen1.5 full finished (7751030), DyTo v3 failed (7751031).

## DyTo Fix History

- **v1** (original): 5.25% — suspected config error
- **v2** (7750932): Copy to /tmp + patch via `/tmp/dyto_fix.py` → FAILED (script missing)
- **v3** (7751031): Pre-patched `/tmp/DYTO_patched` to writable location, then pointed PYTHONPATH at it → ❌ FAILED

## Carya Allocation
79.27% remaining (416,145/525,000 hours)

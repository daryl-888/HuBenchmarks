# Claude's Active Context — 2026-07-22 13:30 CDT

## Current Focus
Both queued jobs resolved: FlashVID qwen1.5 full eval finished (7751030), DyTo v3 failed (7751031). DyTo remains blocked after 3 attempts.

## Repository Structure

```
ovqwen/       Qwen 1.5 — 10 valid models (llava-ov-7b, qwen_1_5)
ovqwen2/      Qwen2   — 15 model dirs (llava-ov-7b-qwen2, qwen_2)
ovqwen3/      Qwen3-VL — 15 model dirs (qwen3-vl-8b, native chat template)
other_backbones/       Non-OV reference (untouched)
experiments/  STTM-LLaVAVid on LLaVA-Video-7B-Qwen2
```

## Carya Disk Space — Resolved
Was 100% full (0MB free), now recovered after other users freed space. No longer blocking.

### Quick Wins (still available):
```bash
rm -rf /project/rhu/dpalfaro/weights/pllava-7b/
rm -rf /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b/
rm -rf /project/rhu/dpalfaro/weights/llava-v1.5-7b/
```
These three unused weight sets (~41GB) can be removed when convenient.

## OVQwen 1.5 Scoreboard (8/10 complete)

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
| 2 | FlashVID (0.15) | **53.29%** | ✅ (7750906) |
| 3= | FlashVID (old 0.25) | **53.36%** | ✅ (7713093) |
| 4 | HoliTom | **53.14%** | ✅ |
| 5 | MDP3 | **53.06%** | ✅ (7713095) |
| 6 | VideoITG | **52.86%** | ✅ (7714855) |
| 7 | AIM | **52.84%** | ✅ (7714969) |
| 8 | FastV | **52.66%** | ✅ (7704758) |
| 9 | STTM-v2 | **51.54%** | ✅ |
| 10 | FastVID | — | ❌ Hard-blocked (0% accuracy) |
| 11 | VisionZip | — | ❌ Broken (LlavaConfig) |

## Recently Resolved Jobs

| Job ID | Model | Backbone | Result |
|:---:|-------|----------|--------|
| 7751030 | FlashVID | qwen1.5 | ✅ Full run finished |
| 7751031 | DyTo | Vicuna-7B | ❌ v3 failed |

## Active Remaining Blockers
- **DyTo**: Failed 3 times (v1: 5.25%, v2: script missing, v3: /tmp patch still failed). LlavaLlamaForCausalLM import error from DYTO/llava module.
- **FastV** (ovqwen1.5): NotImplementedError for Qwen1.5 architecture
- **FastVID**: Model-level bug causing 0% accuracy
- **VisionZip**: LlavaConfig not recognized

## STTM-LLaVAVid Experiments — All 7 Complete
Best config: **t=0.80, 32f → 74.07%** (20/27). STTM adds +14.81% over vanilla.
Recommend full run on t=0.80 config.

## CRITICAL: Carya sbatch Files Are Stale
Fixes applied locally do NOT persist on Carya. Must update Carya sbatch directly.
`scp` always fails (exit 255). Use heredoc to write files directly on Carya:
```bash
ssh carya "cat > /project/rhu/dpalfaro/code/ovqwen/MODEL-motionbenc/run_MODEL.sbatch << 'EOF'
... content ...
EOF"
```
Or use `cat > ...` from local file. Always verify with `grep` after writing.

## Conda Env Reference (on Carya)
| Env | Path | Models |
|-----|------|--------|
| dycoke11 | `/project/rhu/dpalfaro/conda/envs/dycoke11` | DyCoke, AIM, FastVID, FlashVID |
| holitom | `/project/rhu/dpalfaro/conda/envs/holitom` | HoliTom |
| fastv | `/project/rhu/dpalfaro/conda/envs/fastv` | FastV |
| mdp3 | `/project/rhu/dpalfaro/conda/envs/mdp3` | MDP3 (cloned from holitom) |
| sttm_new | `/project/rhu/dpalfaro/conda/envs/sttm_new` | STTM |
| videoitg | `/project/rhu/dpalfaro/conda/envs/videoitg` | VideoITG |
| visionzip | `/project/rhu/dpalfaro/conda/envs/visionzip` | VisionZip (broken) |

## Key Fix Patterns
- **MDP3**: needs `LD_LIBRARY_PATH=/project/rhu/dpalfaro/mdp3_pkgs/torch/lib`, `--pool-frames` (not `--num_frames`), 48h walltime
- **HoliTom**: needs `holitom` env + `PYTHONPATH=HoliTom/Llava-NeXT:HoliTom` + env vars WRAPPER, RETAIN_RATIO, T, HOLITOM_k, HOLITOM_r
- **FastV**: needs `fastv` env + `PYTHONPATH=HoliTom/Llava-NeXT` for llava module
- **DyCoke**: needs `dycoke11` env + `PYTHONPATH=DyCoke`, eval uses `attn_implementation="sdpa"` (not flash_attn)

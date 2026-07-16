# Claude's Active Context — 2026-07-14 12:00 CDT

## Current Focus
Fixing and resubmitting 4 failed OVQwen 1.5 jobs (MDP3, DyCoke, HoliTom, FastV). All 4 have been fixed and resubmitted (7704918-7704921).

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

## Full 8052 Runs — Qwen 1.5 (6 Complete)

See `master-results.md` for full per-category breakdowns.

| # | Model | Accuracy | Run Dir |
|---|-------|:--------:|---------|
| 1 | MDP3 (old) | 53.06% | mdp3_run1 |
| 2 | **AIM** | **52.81%** | ovqwen_aim_run1 |
| 3 | PruneVID | 52.66% | ovqwen_prunevid_run2 |
| 4 | HoliTom (old) | 52.66% | ovqwen_prunevid_run1 (misnamed) |
| 5 | VideoITG (old) | 52.51% | videoitg_run1 |
| 6 | STTM-v2 | 51.87% | ovqwen_sttmv2_run1 |
| | **VideoITG (new)** | **52.86%** | ovqwen_videoitg_run2 (7693609) |

## Currently Queued (7704918-7704921)
| Job ID | Model | Type | Fix Applied |
|:---:|-------|------|-------------|
| 7704918 | MDP3 | Full (48h) | `LD_LIBRARY_PATH` libnccl.so.2 + `--pool-frames` |
| 7704919 | DyCoke | Smoke | Corrupted eval_dycoke.py replaced |
| 7704920 | HoliTom | Full | `holitom` env + PYTHONPATH + WRAPPER/T/k/r env vars |
| 7704921 | FastV | Full | PYTHONPATH to HoliTom/Llava-NeXT (was DyCoke) |

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

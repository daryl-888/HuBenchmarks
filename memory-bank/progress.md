# Progress — HuBenchmarks

## What Works

- **Infrastructure**: SSH access to Carya confirmed working, `sbatch` submissions succeed
- **11 models completed** (accuracy recorded in CLAUDE.md Results Summary):
  - LLaVA-OV-7B baseline (52.69%)
  - LLaVA-Video-7B baseline (56.39%)
  - DyCoke (53.46%)
  - STTM (54.28%)
  - HoliTom (53.11%)
  - VideoITG (52.51%)
  - PruneVid (44.00%)
  - PLLaVA-7B baseline (43.35%)
  - FlashVID (50.50% / 51.99%)
  - FastVID (51.92%)
  - VisionZip (40.09% / 39.97%)
- **NFS stale video patch**: Subprocess-isolated `load_video` deployed in DyCoke's lmms_eval
- **JSONL normalization**: Mixed-type `resolution` field fixed (4,922 rows converted from array to object format)

## What's Left

### 3 models — currently in final debugging, full runs submitted
1. **DyTo** — 0% accuracy on smoke test (vision encoder weights dropped by device_map="auto"). Fixed: pass device_map=None. Resubmitted smoke test as job 7662190.
2. **MDP3** — smoke test PASSED (14/27 = 51.85%). Full run submitted as job 7662191.
3. **AIM** — smoke test PASSED (13/27 = 48.15%). Full run previously failed (stale run_aim.sbatch with old accelerate binary). Fixed: synced correct sbatch. Full run submitted as job 7662192.

### Models still requiring repo setup (no subdirectory yet)
- **iMove**: retrieval-only, no weights released
- **TrajViT**: not runnable

## Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| Conda envs owned by mahern69 (EACCES) | **Ongoing** | pip install --target to dpalfaro-owned dir |
| Stale sbatch on Carya after local edit | **Ongoing** | Sync via heredoc + verify with grep (cost 2+ cycles) |
| DyTo device_map="auto" drops vision weights | **Fixed 2026-07-07** | Pass device_map=None, use model.cuda() |
| NFS stale video handles | **Patched** | Subprocess-isolated load_video with black-frame fallback |
| Mixed-type resolution in JSONL | **Fixed 2026-07-06** | Normalized all 4,922 rows with array format |
| accelerate CLI → register_fake crash | **Workaround** | Use `python -m accelerate.commands.launch` |
| mdp3/aim torch mismatch | **Workaround** | --target to dpalfaro-owned dir, PYTHONPATH + LD_LIBRARY_PATH first |

## Evolution of Project Decisions

- **2026-05**: Initial DyCoke eval — discovered NFS stale video bug → developed subprocess `load_video` patch
- **2026-06-25/30**: STTM, HoliTom, PruneVid, FlashVID, FastVID, VisionZip completed
- **2026-06-30**: DyTo/MDP3/AIM debugging began — discovered stale sbatch sync bug cost 2 job cycles
- **2026-07-04**: All known fixes applied to sbatch files, pushed to git
- **2026-07-06**: Discovered Carya still had stale sbatch files (fixes never synced). Fixed via heredoc. Also discovered JSONL mixed-type resolution bug (AIM env only). All three smoke tests resubmitted.
- **2026-07-07**: DyTo still 0% (device_map deeper issue). MDP3 passed smoke. AIM full run failed (stale run_aim.sbatch). Fixed both. All three jobs resubmitted.

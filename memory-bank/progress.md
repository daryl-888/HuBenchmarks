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

### 3 models — currently in final debugging, jobs submitted
1. **DyTo** — 0% accuracy on 2 smoke tests (vision encoder weights dropped by device_map="auto", then conv_template `image_seq_v3` doesn't exist in DyTo). Fixed: device_map=None, low_cpu_mem_usage=False, vicuna_v1 template. **Smoke test re-submitted as job 7662691.**
2. **MDP3** — smoke test PASSED (14/27 = 51.85%). Full run submitted as job 7662191 (RUNNING, 2h+ elapsed).
3. **AIM** — smoke test PASSED (13/27 = 48.15%). Full run crashed at 542/8052 (KeyError: 'boundaries'). Fixed: kwargs.get('boundaries', None). **Full run re-submitted as job 7662693.**

### Models still requiring repo setup (no subdirectory yet)
- **iMove**: retrieval-only, no weights released
- **TrajViT**: not runnable

## Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| Conda envs owned by mahern69 (EACCES) | **Ongoing** | pip install --target to dpalfaro-owned dir |
| Stale sbatch on Carya after local edit | **Ongoing** | Sync via heredoc/scp + verify with grep (cost 2+ cycles) |
| DyTo device_map="auto" drops vision weights | **Fixed 2026-07-07** | Pass device_map=None, use model.cuda() |
| DyTo conv_template image_seq_v3 missing | **Fixed 2026-07-07** | Use vicuna_v1 (available in DyTo's conversation.py) |
| DyTo low_cpu_mem_usage=True meta-device | **Fixed 2026-07-07** | Patch builder.py to low_cpu_mem_usage=False |
| AIM boundaries KeyError in llava_qwen.py | **Fixed 2026-07-07** | kwargs.get('boundaries', None) |
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
- **2026-07-07**: DyTo still 0% (device_map deeper issue, low_cpu_mem_usage, wrong conv_template). MDP3 passed smoke. AIM full run failed (stale run_aim.sbatch + boundaries KeyError). Fixed all. DyTo test (7662691) and AIM full run (7662693) submitted.
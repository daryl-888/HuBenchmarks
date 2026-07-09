# Active Context — Post-Restructure (2026-07-07)

## Repository Restructured

The repo has been reorganized into backbone-based directories:

```
llava-ov-7b/          LLaVA-OV-7B (Qwen 1.5) — 5 original + 3 ported
llava-ov-7b-qwen2/    LLaVA-OV-7B-Qwen2 (Qwen2) — 3 original + 5 ported
other_backbones/      Non-LLaVA-OV models (STTM, DyTo, PruneVid, VisionZip, etc.)
```

STTM-v2 (which used wrong backbone LLaVA-OV-7B) has been renamed to `other_backbones/incorrect_sttm/`.

## Job Queue

| JobID | Model | Backbone | State | Notes |
|-------|-------|----------|-------|-------|
| 7662788 | DyTo full run | LLaVA-NeXT-Vicuna-7B | PENDING | Test: 3.7% (expected for base model) |
| 7662787 | AIM full run | LLaVA-OV-7B-Qwen2 | PENDING | boundaries fix applied |
| 7662191 | MDP3 full run | LLaVA-OV-7B | RUNNING | 3h40m on compute-10-4 |

## Completed Models (Accuracy)

| Model | Backbone | Accuracy | Directory |
|-------|----------|:--------:|-----------|
| STTM-LLaVAVid | LLaVA-Video-7B | 54.28% | other_backbones/sttm-llavavid-motionbenc/ |
| DyCoke | LLaVA-OV-7B | 53.46% | llava-ov-7b/dycoke-motionbenc/ |
| HoliTom | LLaVA-OV-7B | 53.11% | llava-ov-7b/holitom-motionbenc/ |
| VideoITG | LLaVA-OV-7B | 52.51% | llava-ov-7b/videoitg-motionbenc/ |
| FlashVID | LLaVA-OV-7B-Qwen2 | 51.99% | llava-ov-7b-qwen2/flashvid-motionbenc/ |
| FastVID | LLaVA-OV-7B-Qwen2 | 51.92% | llava-ov-7b-qwen2/fastvid-motionbenc/ |
| STTM-v2 | WRONG BACKBONE | 51.77% | other_backbones/incorrect_sttm/ |
| PruneVid | PLLaVA-7B | 43.80% | other_backbones/prunevid-motionbenc/ |
| VisionZip | LLaVA-1.5-7B | 40.09% | other_backbones/visionzip-motionbenc/ |

## 8 Ported Models Need Verification Runs

Models ported between backbones — sbatch files corrected, not yet run:
- llava-ov-7b/: aim, fastvid, flashvid (ported from Qwen2)
- llava-ov-7b-qwen2/: dycoke, fastv, holitom, mdp3, videoitg (ported from Qwen 1.5)

## Known Issues Still Active

| Issue | Status |
|-------|--------|
| Conda envs owned by mahern69 (EACCES) | Ongoing |
| Stale sbatch on Carya after local edit | Ongoing — must sync via heredoc/scp |
| NFS stale video handles | Patched (subprocess-isolated load_video) |
| DyTo vision encoder weight dropout | Fixed (device_map=None) |
| DyTo conv_template mismatch | Fixed (vicuna_v1) |
| AIM boundaries KeyError | Fixed (kwargs.get) |

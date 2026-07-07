# Active Context — DyTo / MDP3 / AIM Debugging

## Current Status (as of 2026-07-07 12:23)

### Job Queue

| JobID | Model | State | Notes |
|-------|-------|-------|-------|
| **7662788** | DyTo **full run** | PENDING | Test pipeline (7662772) ran cleanly with 3.7% accuracy (expected for base model). Full run with actual weights submitted. |
| **7662787** | AIM **full run** | PENDING | Previous (7662693) crashed at sample 542/8052: `TypeError: object of type 'NoneType' has no len()`. Fix: `kwargs.get('boundaries', [])` instead of `None`. |
| **7662191** | MDP3 **full run** | RUNNING | 3h40m on compute-10-4. |

### Root Causes Fixed

**DyTo** — 4 patches applied to `llava_arch.py`:
1. `image_features` was a **list** of per-frame tensors when `mm_patch_merge_type='flat'`. `temporal_aggregation` called `.shape` on it → crash. **Fix**: stack with `torch.stack()` before calling temporal_aggregation.
2. `FINCH()` call had `tw_finch=True` kwarg that the installed `finch-clust` library doesn't support. **Fix**: removed the kwarg.
3. `finch_cluster` method **never returned anything** — computed `selected_indexs` but never selected embeddings or returned. **Fix**: added embedding selection + stack + return.
4. `image_sizes` was a list of PIL `(w,h)` tuples → model expects tensor of `(h,w)`. **Fix**: convert to `torch.tensor([[h,w],...])`.

**AIM** — 2 fixes:
1. `kwargs['boundaries']` KeyError → `kwargs.get('boundaries', [])` (was `None`, but Qwen2 forward calls `len(boundaries)` so needs empty list)
2. `accelerate` CLI binary crash → `python -m accelerate.commands.launch`

**MDP3** — Still running, handling NFS stale handles gracefully.
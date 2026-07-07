# Active Context — DyTo / MDP3 / AIM Debugging

## Current Status (as of 2026-07-07)

Three jobs submitted and PENDING on Carya. MDP3 smoke test passed (51.85%). AIM and DyTo had recurring stale-sbatch issues that are now fixed.

### Job Queue

| JobID   | Model | What                | State   | Notes |
|---------|-------|---------------------|---------|-------|
| 7662206 | DyTo  | test (low_cpu_mem_usage=False) | PENDING | Previous test still 0% — builder.py's low_cpu_mem_usage=True causes meta-device even with device_map=None |
| 7662191 | MDP3  | full run            | RUNNING | 7+ hrs on compute-10-4. Smoke test passed 14/27 = 51.85% |
| 7662192 | AIM   | full run            | RUNNING | 4+ hrs on compute-9-3. Smoke test passed 13/27 = 48.15% |

### What to do when jobs finish

```bash
sacct -j 7662206,7662191,7662192 --format=JobID,JobName,State,ExitCode -X
# DyTo test accuracy:
grep 'Accuracy:' /project/rhu/dpalfaro/results/dyto_test_7662206.out
# MDP3 full accuracy:
cat /project/rhu/dpalfaro/results/mdp3_run1/summary.json | grep -E 'accuracy|correct|total'
# AIM full accuracy:
cat /project/rhu/dpalfaro/results/aim_run1/summary.json | grep -E 'accuracy|correct|total'
# Then record in CLAUDE.md Results Summary table
```

### Known Issues — Complete Fix History Per Model

**DyTo** (5 layers peeled):
1. PYTHONPATH needed both DYTO/ and DYTO/dyto → fixed in sbatch
2. builder.py's `from llava.model import *` doesn't export LlavaLlamaForCausalLM → patch needed
3. finch-clust missing from conda env → installed to `/project/rhu/dpalfaro/dyto_pkgs` via `--target`
4. `device_map="auto"` (builder default) silently drops vision encoder weights → passed `device_map=None` + `model.cuda()` in eval_dyto.py
5. `low_cpu_mem_usage=True` (hardcoded in builder.py) causes meta-device regardless of device_map → patched to `low_cpu_mem_usage=False` directly in `/project/rhu/dpalfaro/DYTO/dyto/llava/model/builder.py` via sed

**MDP3** (3 layers peeled):
1. conda env cloned from holitom inherits broken torch 2.6.0→2.3.1 (missing libcudart.so.12)
2. torch 2.3.1+cu121 installed to `/project/rhu/dpalfaro/mdp3_pkgs` via `pip install --target`; sbatch needs `module load cudatoolkit/12.1` + `LD_LIBRARY_PATH`
3. `libnccl.so.2` not in torch wheel — conda env has it at `nvidia/nccl/lib/` — added to `LD_LIBRARY_PATH`

**AIM** (3 fixes):
1. `accelerate` CLI binary crashes (timm→torchvision register_fake) → use `python -m accelerate.commands.launch`
2. conda env torch/torchvision incompatible → matched pair installed to `/project/rhu/dpalfaro/aim_pkgs`
3. run_aim.sbatch on Carya was stale (old `~/.local` + `accelerate` binary) while test_aim.sbatch was correct — synced both

### Critical Operating Rule

NEVER assume a fix "took effect" on Carya just because you edited/committed a file locally. Always sync via heredoc or scp and verify with `grep` on the remote file. This mistake happened **twice** in this debugging session: evaluated fixes were not synced to Carya in two separate batches.

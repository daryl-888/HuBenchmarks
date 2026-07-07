# Active Context — DyTo / MDP3 / AIM Debugging

## Current Status (as of 2026-07-07 11:23)

Three jobs submitted on Carya. MDP3 still running. DyTo smoke test re-failed (0%) — conv_template was wrong. AIM full run crashed (boundaries KeyError).

### Fixes Applied
1. **DyTo**: Changed `--conv-template` from `image_seq_v3` (doesn't exist in DyTo's conversation.py) to `vicuna_v1` (available). Also moved sbatch + eval paths from `/project/rhu/dpalfaro/code/dyto-motionbenc/` (mahern69-owned, EACCES) to `/project/rhu/dpalfaro/dyto-motionbenc/` (dpalfaro-owned).
2. **AIM**: Patched `llava_qwen.py` line 143 — changed `kwargs['boundaries']` to `kwargs.get('boundaries', None)` to fix KeyError when `boundaries` is not passed in subsequent generation steps.

### Job Queue

| JobID | Model | What | State | Notes |
|-------|-------|------|-------|-------|
| **7662691** | DyTo | test (vicuna_v1 conv_template) | PENDING | Previous test: 0/27 = 0.0000. Fixed conv_template from image_seq_v3 to vicuna_v1 |
| **7662693** | AIM | full run (boundaries fix) | PENDING | Previous run: COMPLETED but crashed at 542/8052 with KeyError 'boundaries'. Fixed via sed on Carya |
| **7662191** | MDP3 | full run | RUNNING | 2h 39m on compute-10-4. Partial results in results.jsonl. Smoke test: 51.85% |

### Known Issues — Complete Fix History Per Model

**DyTo** (6 layers peeled):
1. PYTHONPATH needed both DYTO/ and DYTO/dyto → fixed in sbatch
2. builder.py's `from llava.model import *` doesn't export LlavaLlamaForCausalLM → patch needed
3. finch-clust missing from conda env → installed to `/project/rhu/dpalfaro/dyto_pkgs` via `--target`
4. `device_map="auto"` (builder default) silently drops vision encoder weights → passed `device_map=None` + `model.cuda()` in eval_dyto.py
5. `low_cpu_mem_usage=True` (hardcoded in builder.py) causes meta-device regardless of device_map → patched to `low_cpu_mem_usage=False` directly in builder.py
6. **`image_seq_v3` conv_template doesn't exist in DyTo's conversation.py** — only vicuna v0/v1, llava v0/v1, etc. Changed to `vicuna_v1`. **Fixed 2026-07-07**

**MDP3** (3 layers peeled):
1. conda env cloned from holitom inherits broken torch 2.6.0→2.3.1 (missing libcudart.so.12)
2. torch 2.3.1+cu121 installed to `/project/rhu/dpalfaro/mdp3_pkgs` via `pip install --target`; sbatch needs `module load cudatoolkit/12.1` + `LD_LIBRARY_PATH`
3. `libnccl.so.2` not in torch wheel — conda env has it at `nvidia/nccl/lib/` — added to `LD_LIBRARY_PATH`

**AIM** (4 fixes):
1. `accelerate` CLI binary crashes (timm→torchvision register_fake) → use `python -m accelerate.commands.launch`
2. conda env torch/torchvision incompatible → matched pair installed to `/project/rhu/dpalfaro/aim_pkgs`
3. run_aim.sbatch on Carya was stale (old `~/.local` + `accelerate` binary) while test_aim.sbatch was correct — synced both
4. **`kwargs['boundaries']` KeyError in `llava_qwen.py` line 143** — on second generation step, `boundaries` is not passed in kwargs. Changed to `kwargs.get('boundaries', None)`. **Fixed 2026-07-07**

### Critical Operating Rule

NEVER assume a fix "took effect" on Carya just because you edited/committed a file locally. Always sync via heredoc or scp and verify with `grep` on the remote file. This has been the root cause of multiple wasted job cycles.
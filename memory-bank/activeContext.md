# Active Context — DyTo / MDP3 / AIM Debugging

## Current Status (as of 2026-07-06)

Three models are in the final debugging phase. All smoke tests have been submitted and are PENDING on Carya.

### Job Queue

| JobID   | Model | Name     | State   | Reason        |
|---------|-------|----------|---------|---------------|
| 7660344 | DyTo  | dyto_test| PENDING | Resources     |
| 7660345 | MDP3  | mdp3_test| PENDING | Priority      |
| 7660346 | AIM   | aim_test | PENDING | Priority      |

### What to do when jobs finish

```bash
sacct -j 7660344,7660345,7660346 --format=JobID,JobName,State,ExitCode -X
# Check errors:
cat /project/rhu/dpalfaro/results/dyto_test_7660344.err
cat /project/rhu/dpalfaro/results/mdp3_test_7660345.err
cat /project/rhu/dpalfaro/results/aim_test_7660346.err
```

If any fail with NEW errors (different from the tracebacks already fixed), that means another layer of the problem has been exposed. Each fix peels back one error to reveal the next.

If all succeed → submit full runs (no `--limit`):
- `sbatch /project/rhu/dpalfaro/test_dyto.sbatch` → switch to `run_dyto.sbatch`
- `sbatch /project/rhu/dpalfaro/code/mdp3-motionbenc/test_mdp3.sbatch` → switch to `run_mdp3.sbatch`
- `sbatch /project/rhu/dpalfaro/aim-motionbenc/test_aim.sbatch` → switch to `run_aim.sbatch`

Then record accuracy in Results Summary table in `CLAUDE.md`.

### Known Issues Already Fixed (per model)

**DyTo**:
1. PYTHONPATH needed both DYTO/ and DYTO/dyto
2. builder.py's `from llava.model import *` doesn't export LlavaLlamaForCausalLM → patch needed
3. finch-clust missing from conda env → installed to `/project/rhu/dpalfaro/dyto_pkgs` via `--target`

**MDP3**:
1. conda env cloned from holitom inherits broken torch 2.6.0→2.3.1 (missing libcudart.so.12)
2. torch 2.3.1+cu121 installed to `/project/rhu/dpalfaro/mdp3_pkgs` via `pip install --target`
3. sbatch needs `module load cudatoolkit/12.1` + `LD_LIBRARY_PATH` for torch libs

**AIM**:
1. `accelerate` CLI binary crashes (timm→torchvision register_fake) → use `python -m accelerate.commands.launch`
2. conda env torch/torchvision incompatible → matched pair installed to `/project/rhu/dpalfaro/aim_pkgs`
3. pyarrow 24.0.0 + datasets 2.16.1 chokes on mixed-type `resolution` field → normalized JSONL

### Critical Operating Rule

NEVER assume a fix "took effect" on Carya just because you edited/committed a file locally. Always sync via heredoc or scp and verify with `grep` on the remote file.

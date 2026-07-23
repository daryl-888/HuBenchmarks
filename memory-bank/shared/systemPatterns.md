# System Patterns — HuBenchmarks

## Architecture

Models are organized by backbone into three top-level directories:

```
llava-ov-7b/              # LLaVA-OV-7B (Qwen 1.5) — 5 original + 3 ported
llava-ov-7b-qwen2/        # LLaVA-OV-7B-Qwen2 (Qwen2) — 3 original + 5 ported
other-backbones/          # Non-LLaVA-OV models (STTM, DyTo, PruneVid, VisionZip, etc.)
```

Each model is a self-contained subdirectory:

```
<model>-motionbenc/
  test_<model>.sbatch     # smoke test (limit varies)
  run_<model>.sbatch      # full evaluation (no limit)
  run_baseline.sbatch     # vanilla backbone comparison
  eval_<model>.py         # custom evaluation script (or uses lmms_eval)
  tasks/motionbench/
    motionbench.yaml      # lmms_eval task definition
    utils.py              # doc_to_visual, doc_to_text, process_results, aggregate
  strict/                 # strict evaluation variants
  PORTED.md               # present if ported from other backbone (needs verification run)
```

## Key Design Patterns

### 1. sbatch-as-documentation pattern
Each sbatch file contains both the SLURM directives AND a commented "One-Time Setup" section that documents every step needed to prepare the env. This means:
- Any user can reconstruct the env from the sbatch alone
- The setup instructions are version-controlled alongside the submission script
- No hidden state in shell history or README-only docs

### 2. pip install --target workaround
Many conda envs on Carya are owned by `mahern69` (EACCES for dpalfaro). The standard workaround:
```bash
pip install --target /project/rhu/dpalfaro/<model>_pkgs <package>
```
Then put that directory FIRST on PYTHONPATH in the sbatch.

### 3. Subprocess-isolated load_video
NFS stale handles cause D-state kernel hangs that can't be caught by try/except. The fix:
- Launch video loading in a `multiprocessing.Process`
- `Queue.get(timeout=60)` for the result
- `proc.kill()` on timeout (D-state ignores SIGKILL but at least don't wait forever)
- Return black frames instead of raising

### 4. Stale sbatch sync protocol
Git push does NOT update Carya. Required sync steps:
1. Edit locally → commit + push to `claude/bold-knuth-YlJYI`
2. `ssh dpalfaro@carya... "cat > file << 'EOF'"` (heredoc) OR `scp`
3. Verify: `ssh ... "grep <key_pattern> /path/to/file"`

### 5. Debugging by peeling layers
Each fix reveals the next error. Pattern:
1. Submit smoke test → check `.err` for traceback
2. Fix the specific error found
3. Sync fix to Carya
4. Resubmit → check for NEW error (different from previous)
5. Repeat until job completes successfully

## Component Relationships

```
User → sbatch → SLURM → GPU node
                  ↓
         ./eval_<model>.py  (or lmms_eval)
                  ↓
         MotionBench dataset (/project/rhu/MotionBench_Data)
                  ↓
         Results → /project/rhu/dpalfaro/results/

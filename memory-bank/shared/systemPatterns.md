# System Patterns — HuBenchmarks

## Architecture

Models are organized by backbone generation (restructured 2026-07-23):

```
stage1-llava-ov/          # LLaVA-OV-7B  (mid-2024)   — 10 methods
stage3-qwen3-vl/          # Qwen3-VL-8B  (late-2025)  — 10 methods + baseline
other-backbones/          # each method's own native model (DyTo/Vicuna,
                          #   PruneVid/PLLaVA, VisionZip/LLaVA-1.5)
archive/                  # descoped Stage 2, legacy sbatch, superseded notes
```

**The `llava-ov-7b` vs `llava-ov-7b-qwen2` split was a misconception** — both
names referred to the *same weights* (`LlavaQwenForCausalLM`, Qwen2 internally).
`qwen_1_5` vs `qwen_2` control prompt formatting only. The duplicate weights
directory was deleted; do not reintroduce the distinction.

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
  smoke_<model>*.sbatch   # 8-sample gate check before any full run
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
1. Edit locally → commit
2. `./scripts/deploy.sh --push <subtree>` (rsync; **`--push` is required — the
   default is a dry run** and prints "DRY RUN complete" if you forget)
3. Verify: `ssh ... "grep <key_pattern> /path/to/file"`

`scripts/` is not in the deploy subtrees; copy those with `scp` to
`/project/rhu/dpalfaro/code/HuVLLM_scripts/`.

### 5. Debugging by peeling layers
Each fix reveals the next error. Pattern:
1. Submit smoke test → check `.err` for traceback
2. Fix the specific error found
3. Sync fix to Carya
4. Resubmit → check for NEW error (different from previous)
5. Repeat until job completes successfully

### 6. Two-signal verification — the rule everything else rests on

A method is only "verified" when **both** hold:

1. it logs `<METHOD> ACTIVE: ...` with real numbers (the mechanism ran), and
2. its predictions **differ from the plain backbone** (it changed the output).

Signal 1 alone is worthless. Three separate ports printed ACTIVE (or completed
cleanly) while producing byte-identical output to the backbone — FastV/LLaVA-OV,
PruneVID/LLaVA-OV, STTM/Qwen3-VL — and only the divergence check caught them.
Every full run is smoke-gated at 8 samples with `--vs-baseline` first.

Corollary: **matching accuracy is never evidence.** `flashvid_run4` tied DyCoke's
score exactly while differing on 1,240/8,052 predictions.

### 7. Numbers carry their caveats in the data, not just the prose

Where a result is a partial or a reconstruction, `summary.json` records it
(`finch_variant`, `paper_faithful: false`, `report_as: ...`). A caveat that lives
only in a markdown file gets separated from the number it qualifies.

## Component Relationships

```
User → sbatch → SLURM → GPU node
                  ↓
         ./eval_<model>.py  (or lmms_eval)
                  ↓
         MotionBench dataset (/project/rhu/MotionBench_Data)
                  ↓
         Results → /project/rhu/dpalfaro/results/

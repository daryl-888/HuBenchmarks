# scripts/

## deploy.sh — local → Carya sync (kills the "stale sbatch" trap)

`git push` does **not** update Carya. Every time you edit an sbatch / utils.py /
eval script locally, you must push it to the cluster or the job runs the OLD file
and fails identically. This script does that push in one command.

```bash
scripts/deploy.sh                    # dry run — show what would change, touch nothing
scripts/deploy.sh --push             # sync all four wrapper trees
scripts/deploy.sh --push stage1-llava-ov               # one tree
scripts/deploy.sh --push stage1-llava-ov/dycoke-motionbenc   # one wrapper
```

**Workflow rule:** after ANY local edit to a wrapper, run
`scripts/deploy.sh --push <that wrapper>` *before* `sbatch`. If in doubt, dry-run
first — it lists exactly which files differ.

### What it does / doesn't touch

- Syncs only: `stage1-llava-ov/`, `stage2-llava-video/`, `stage3-qwen3-vl/`,
  `other-backbones/` — the eval wrappers, mapped 1:1 to the same names under
  `/project/rhu/dpalfaro/code/`.
- **Never** touches the method source forks (`DyCoke/`, `HoliTom/`, `MDP3/`,
  `LLaVA/`, …) or `weights/`. Those are hand-managed and often contain
  Carya-only patches (e.g. the NFS `load_video` fix in CLAUDE.md).
- Uses `rsync` if present locally; otherwise falls back to a `tar | ssh` pipe
  (Linux Mint often ships no rsync). Install rsync for true mirroring incl.
  deletion: `sudo apt install rsync`.

## check_run.py — sanity gate (turns silent-wrong into loud FAIL)

The most expensive failure mode here is a run that *completes and lies* — a
plausible accuracy from a method that never actually engaged. Real cases this
project hit: FastV ran as a pure baseline (`enabled: false`) but was recorded as
"FastV"; VisionZip produced 0% empty predictions; PruneVID dropped weights and
scored 44%. Each looked like data, not a bug.

`check_run.py` reads the two files every eval writes (`summary.json`,
`results.jsonl`) and FAILs on any known silent-failure signature: truncated run,
NA-count drift, accuracy outside [0.40, 0.95], empty/constant predictions, or a
`*_params.enabled == false` method-no-op.

```bash
python scripts/check_run.py <run_dir> [--expect-method NAME] [--strict]
# on Carya:
python code/HuVLLM_scripts/check_run.py /project/rhu/dpalfaro/results/<run> --expect-method dycoke
```

Exit 0 = trustworthy; non-zero = do not record. `--strict` also fails on warnings.

**Automate it:** paste `gate_snippet.sh` at the end of each eval sbatch (after the
python eval call). It runs the gate and renames the output dir to
`*.FAILED_GATE` if the run is untrustworthy, so a bad run can never be silently
mistaken for a good one.

### Known second half of the trap: leftover flat dirs

Carya still has the **pre-restructure flat wrappers** alongside the new stage
layout, e.g. `/code/dycoke-motionbenc/` next to
`/code/stage1-llava-ov/dycoke-motionbenc/`. Some sbatch files still `cd` into the
flat path. If a job ignores your edits, check which path its sbatch actually
references — you may be editing the stage copy while the job runs the flat copy.
Long-term fix: repoint all sbatch files at the stage paths and delete the flat
dirs on Carya.

# DyTo — Unmodified Upstream Verification Run

**Status: verification only.** This folder runs DyTo's code **exactly as released
upstream** — zero patches to any DyTo file. The only code we add is the wrapper
that (a) converts MotionBench metadata into the GT-JSON format DyTo's own
scripts expect, and (b) invokes DyTo's own `run_inference.py` entry point
untouched.

## Why this exists

The gated number in the tables (`results-cache/ob_dyto_run`, 42.06%) was
produced from a **patched** DyTo tree:

- `finch_cluster()` had a missing `return` transcribed back in
  (`patches/dyto/finch_cluster_return.patch`)
- The `tw_finch=` kwarg, absent from the pinned `finch-clust==0.2.0`, was
  shimmed via our reconstructed TW-FINCH (`reconstruction_notes` in the summary)

`docs/UPSTREAM_DEFECTS.md` documents both as **defects in the released code**.
This run re-verifies that claim against the pristine upstream tree: it attempts
the identical evaluation with **no source edits whatsoever**, so the run is
expected to terminate at the first defect with a real traceback. That failure
*is* the verification.

## What exactly is run (nothing patched)

```
DYTO/  (fresh `git clone https://github.com/Jam1ezhang/DYTO`, pinned commit)

DYTO/run_inference.py          -- DyTo's own orchestrator (untouched)
DYTO/run_inference_multiple_choice_qa.py  -- DyTo's own MCQA entry (untouched)
DYTO/dataset.py, prompt.py, utils.py       -- DyTo's own helpers (untouched)
DYTO/dyto/**                   -- DyTo's LLaVA fork (untouched)
```

The wrapper (`eval_dyto_official.py`) does **not** import `dyto.*` and does
**not** call `model.generate(...)`. It only:

1. reads `$MOTIONBENCH/video_info.meta.jsonl`
2. converts each row to DyTo's GT-JSON schema (`task_name`, `video_name`,
   `question_id`, `question`, `candidates`, `answer_number`, `answer`)
3. writes the DyTo-style `exp_config` YAML (`SCRIPT=run_inference_multiple_choice_qa.py`)
4. runs `python run_inference.py --exp_config <cfg>` as a subprocess
5. if the official script survives (it should not, per the defect analysis),
   scores its JSON output with the repo's standard letter/NA protocol

## Expected outcome

| Stage | Expectation | Evidence |
|---|---|---|
| Model load | runs (Vicuna-7B, conv `vicuna_v1`, `rope_scaling_factor=2`) | `README.md` quick start |
| First video, `temporal_aggregation=spatial_tome_finch_dynamic_all_frms` | **crash**: `FINCH() got an unexpected keyword argument 'tw_finch'` | `docs/UPSTREAM_DEFECTS.md` §1.1, pinned `finch-clust==0.2.0` |
| If `tw_finch` were bypassed | next crash: `'NoneType' object has no attribute 'shape'` at `llava_arch.py:234` (missing `return` in `finch_cluster()`) | `docs/UPSTREAM_DEFECTS.md` §1.2 |

The wrapper never patches around either defect. No predictions are expected.

## Layout

| File | Role |
|---|---|
| `verify_upstream.sh` | Clone + pin upstream; assert tree hash; **no patches** |
| `prepare_dyto_gt.py` | MotionBench meta → DyTo GT JSON + exp_config YAML |
| `eval_dyto_official.py` | Orchestrates DyTo's own `run_inference.py`; captures traceback; scores if any output |
| `run_dyto_official.sbatch` | Full 8,052-sample verification job |
| `test_dyto_official.sbatch` | Smoke (`--limit 8`) verification job |
| `README.md` | This file |

## Setup on Carya

> **Deploy note.** `git push` does NOT update Carya, and `scripts/deploy.sh`
> only syncs the four sanctioned wrapper trees (`stage1-llava-ov`,
> `stage2-llava-video`, `stage3-qwen3-vl`, `other-backbones`) — `analysis/` is
> **not** among them. So this folder must be shipped to the cluster explicitly.

```bash
# 0. Deploy this folder to Carya (run from the LOCAL repo root):
rsync -az --itemize-changes \
    analysis/upstream-faithful/dyto-official/ \
    dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/analysis/upstream-faithful/dyto-official/
#    (or, if your machine has no rsync:)
#    scp -r analysis/upstream-faithful/dyto-official \
#        dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/analysis/upstream-faithful/

# 1. Clone + pin upstream (no patches), ON CARYA:
bash /project/rhu/dpalfaro/code/analysis/upstream-faithful/dyto-official/verify_upstream.sh

# 2. Weights (already present)
#    /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b

# 3. Env (DyTo's own pins: torch==2.2.0, transformers==4.38.2, finch-clust==0.2.0)
conda create -n dyto_v python=3.10 -y
/project/rhu/dpalfaro/conda/envs/dyto_v/bin/pip install \
    torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121
/project/rhu/dpalfaro/conda/envs/dyto_v/bin/pip install \
    -e /project/rhu/dpalfaro/code/analysis/upstream-faithful/dyto-official/DYTO
/project/rhu/dpalfaro/conda/envs/dyto_v/bin/pip install \
    finch-clust==0.2.0 decord

# 4. Smoke
sbatch /project/rhu/dpalfaro/code/analysis/upstream-faithful/dyto-official/test_dyto_official.sbatch
```

## Consensus — re-verified against git history + committed bytecode

Re-inspecting the live upstream (`github.com/Jam1ezhang/DYTO`, HEAD
`570e977` 2025-10-25) against its own git history yields a consensus:

**The released code is genuinely incorrect — it cannot run as published —
but the evidence shows this is a release-hygiene / dirty-commit problem,
not a deliberate misstatement.**

Evidence:

| # | Fact (verified) | Implication |
|---|---|---|
| 1 | `finch_cluster()` (HEAD) has **zero `return` statements**; ends at the `else:` block; caller at line 233 dereferences the `None` | Released code crashes with `'NoneType' object has no attribute 'shape'` on the **official primary config** |
| 2 | The ONLY two commits both call a function that never existed in source: commit `2c6d0f7` calls `self.finch_clusterv2(...)` (undefined → `NameError`); commit `ae46634` renamed the call to `self.finch_cluster(...)` (the return-less one) | The call site has never matched a defined, returning function **in any published revision** |
| 3 | The committed `dyto/llava/model/__pycache__/llava_arch.cpython-310.pyc` (authors accidentally committed `__pycache__`) contains the code object **`LlavaMetaForCausalLM.finch_clusterv2` ending in `RETURN_VALUE`** | The file the authors **actually ran** during development had a *working* `finch_clusterv2` that returned a tensor. The source they committed is a different, incomplete edit |
| 4 | The official 7B config uses `TEMPORAL_AGGREGATION: "spatial_tome_finch_dynamic_all_frms"` (the broken path) and `SCRIPT: bash scripts/run_eval_*.sh` — but **no `scripts/` directory exists anywhere in the repo** | Runnable-version runtime artifacts were never published |
| 5 | `finch-clust==0.2.0` (PyPI) has no `tw_finch` parameter; live code calls `FINCH(..., tw_finch=tw_finch)` | TW-FINCH was a local modification that was never published |

**Verdict.** The paper's results were produced with a private working tree that
(per the bytecode) had a returning `finch_clusterv2`, a FINCH with `tw_finch`,
and the `scripts/` harness. The published repository is an incomplete snapshot:
the `.pyc` leaks the working version, the source does not. That is *sloppy
release hygiene / dirty working tree*, not a deliberate attempt to ship broken
code — there is no plausible way to run the committed source end-to-end, which
is what the verification job in this folder demonstrates.

This supports (rather than contradicts) `docs/UPSTREAM_DEFECTS.md` §1.1/§1.2,
and adds the mechanism: **"the authors committed a different (older) revision
of `llava_arch.py` than the one that produced their numbers."**

## How to read the result

- **FAIL with `tw_finch` traceback** → confirms `UPSTREAM_DEFECTS.md` §1.1
  against the pristine tree. Verification **passed** (the defect reproduced).
- **FAIL with `NoneType ... shape`** → §1.1 got past (e.g. newer upstream fix),
  §1.2 reproduced. Still a pass of the overall verification.
- **Any predictions produced** → the upstream tree changed; re-open the defect
  analysis before trusting the number.
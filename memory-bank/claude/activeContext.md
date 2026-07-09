# Claude's Active Context — Post-Standardization (2026-07-09)

## Repository Structure

```
ovqwen/       Qwen 1.5 — 8 OV models + sttm (native) + 6 NOG
ovqwen2/      Qwen2   — 8 OV models + 7 NOG
ovqwen3/      Qwen3   — 8 OV models (template) + 7 NOG
other_backbones/       Non-OV reference (untouched)
```

## Standardized Eval Pattern

All models share a common eval script at `scripts/eval_template.py`. The standard:
- 32 frames per video (uniform sampling)
- Subprocess-isolated video loading (NFS safety)
- Greedy decoding: `do_sample=False, max_new_tokens=16`
- Letter-match scoring (A-D regex, NA-skip)
- Output: `results.jsonl` + `summary.json` with `per_category` breakdown

### Models with standalone stubs (needs model-loading filled by other person)

| Model | Folders | Eval Script | Status |
|-------|---------|-------------|--------|
| aim | ovqwen, ovqwen2, ovqwen3 | eval_aim.py | Stub — needs AIM's token merge + prune loader |
| dycoke | ovqwen, ovqwen2, ovqwen3 | eval_dycoke.py | Stub — needs DyCoke's KV cache compression loader |
| fastvid | ovqwen, ovqwen2, ovqwen3 | eval_fastvid.py | Stub — needs FastVID's DySeg+STPrune+DTM loader |
| flashvid | ovqwen, ovqwen2, ovqwen3 | eval_flashvid.py | Stub — needs FlashVID's wrapper loader |

### Models with working standalone scripts (already complete)

| Model | Folders | Eval Script | Status |
|-------|---------|-------------|--------|
| holitom | ovqwen, ovqwen2, ovqwen3 | eval_holitom.py | ✅ Complete |
| mdp3 | ovqwen, ovqwen2, ovqwen3 | eval_mdp3.py | ✅ Complete |
| fastv | ovqwen, ovqwen2, ovqwen3 | eval_fastv.py | ✅ Complete |
| videoitg | ovqwen, ovqwen2, ovqwen3 | eval_videoitg_infer.py | ✅ Complete (two-stage) |
| sttm | ovqwen | eval_sttm.py | ✅ Complete (+ NOG copies) |

## Fixes Applied (2026-07-09)

1. **AIM conv_template**: `qwen_1_5` → `qwen_2` (all 3 folders, run + test sbatch) — confirmed by paper audit
2. **FlashVID 8f → 32f**: `--frame-counts 8` → `--num_frames 32`, removed `--prompt-style strict`
3. **12 eval stubs created** from standardized template — `load_model()` is a stub, needs filling in

## Results Gatherer

Use `python analysis/gather_results_v2.py` to scan all folders and produce results table.
Options: `--per-category`, `--json`, `--csv output.csv`

## Open Issues

| Priority | Action | Owner |
|:--------:|--------|:-----:|
| 🔴 HIGH | Fill in `load_model()` stubs in 12 eval scripts | Other person |
| 🟡 MEDIUM | Run smoke tests on ovqwen2 ported models | Either |
| 🟡 MEDIUM | Verify FlashVID 32f works with new eval | Either |
| ℹ️ LOW | Update Carya paths to reflect new ovqwen* structure | Either |

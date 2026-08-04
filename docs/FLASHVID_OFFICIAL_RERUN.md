# Re-running FlashVID on Qwen3-VL against the authors' own code

**Status: in progress.** Two independent bugs blocked the run before any
prediction could be scored; both are documented here as they were found and
fixed. This page will be folded into [UPSTREAM_CROSSREF.md](UPSTREAM_CROSSREF.md)
once the run gates.

## Why this run exists

`w3_flashvid_run` (56.65%, the number in the results tables) never imports
FlashVID's own package. It reimplements the token-scoring rule and omits
inner-LLM compression (`pruning_layer=28`, `llm_retention_ratio=0.1`),
`expansion=1.25`, `min_segment_num=4`, `complementary_segment`,
`segment_threshold` and `token_selection_method=attn_div`. FlashVID ships
native Qwen3-VL support — `flashvid/modeling_qwen3_vl.py` plus
`scripts/qwen3_vl.sh` — so `analysis/upstream-faithful/eval_flashvid_qwen3vl_official.py`
calls the authors' `flashvid()` directly with those settings, on Qwen3-VL-8B.

## Bug 1 — dtype crash, the authors' regression

First full run (job 7961637, 2026-08-03): **0.00% accuracy, 100% empty
predictions**, `expected scalar type Float but found BFloat16` on 8,044 of
8,052 samples, from sample 0 onward. The gate caught it correctly
(`RESULT: FAIL`) — this is exactly the failure mode the gate exists to catch,
not a case it missed.

Traced to `flashvid/modeling_qwen3_vl.py`'s `Qwen3VLVisionModel_forward`:

```python
hidden_states = self.patch_embed(hidden_states)
pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
hidden_states = hidden_states + pos_embeds          # <- no dtype cast
```

Stock `transformers` (`models/qwen3_vl/modeling_qwen3_vl.py:703`) has:

```python
hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
```

FlashVID's patched forward was copied from an upstream version and dropped
the `.to(hidden_states.dtype)` cast. `fast_pos_embed_interpolate` returns
float32 (interpolation on the position-embedding buffer); with the model
loaded in bf16, the unguarded add fails outright. **This is a bug in the
released FlashVID code**, not an artifact of our harness or environment —
confirmed by diffing against the stock implementation it was patched from.

Fix applied directly to the vendored clone,
`/project/rhu/dpalfaro/code/FlashVID/flashvid/modeling_qwen3_vl.py`
(one line, `.to(hidden_states.dtype)` restored), with the original backed up
alongside it. Re-running the same sample after the fix advanced past this
point cleanly (0 `BFloat16` warnings).

## Bug 2 — hard FlashAttention-2 requirement, no matching build on this cluster

With bug 1 fixed, every sample still failed, now with an *empty* exception
message. A minimal repro script (bypassing the eval loop's broad
`except Exception`) surfaced the real traceback:

```
File ".../flashvid/modeling_qwen3_vl.py", line 55, in Qwen3VLVisionAttention_forward
    assert self.config._attn_implementation == "flash_attention_2"
AssertionError
```

FlashVID's vision attention has no `sdpa` or `eager` fallback — it requires
FlashAttention-2, unconditionally. The `qwen3vl` conda env had no `flash_attn`
package installed at all.

**Attempt 1 — prebuilt wheel.** `torch==2.6.0+cu124`, Python 3.10, cxx11abi
False. Fetched the matching asset from `Dao-AILab/flash-attention`'s GitHub
releases (`flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`).
Installed cleanly, but failed to import:

```
ImportError: flash_attn_2_cuda...so: undefined symbol:
_ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

A libtorch C++ ABI mismatch not captured by the wheel's coarse `torch2.6` tag
— the same symbol error reproduced across all `torch2.6`-tagged releases
(v2.8.0.post2 through v2.8.3.post1) checked.

**Attempt 2 — source build via `pip install --no-build-isolation`.** The
cluster has a matching `CUDA/12.4.0` module (`nvcc`) and a local `ninja`.
Submitted as a job with that module loaded. It "succeeded" in seconds and
produced the identical broken artifact — flash-attn's `setup.py` has a
shortcut that guesses a prebuilt-wheel URL matching the detected torch
version and downloads it instead of compiling, silently, unless
`FLASH_ATTENTION_FORCE_BUILD=TRUE` is set. The build log's own line gave it
away: `Guessing wheel URL: ...v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.6...` —
the same wheel as attempt 1.

**Attempt 3 — forced source build.** Resubmitted with
`FLASH_ATTENTION_FORCE_BUILD=TRUE`, `MAX_JOBS=6`, `CUDA/12.4.0` loaded,
5-hour limit. In progress.

## Reproduce

```bash
# the fix (already applied on Carya)
diff /project/rhu/dpalfaro/code/FlashVID/flashvid/modeling_qwen3_vl.py.bak \
     /project/rhu/dpalfaro/code/FlashVID/flashvid/modeling_qwen3_vl.py

# the build
sbatch analysis/upstream-faithful/build_flash_attn.sbatch

# the run, once flash-attn imports cleanly
sbatch analysis/upstream-faithful/run_flashvid_qwen3vl_official.sbatch
```

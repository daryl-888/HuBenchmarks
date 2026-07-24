# Defects Found in Published Method Code

Reproducing eleven training-free video-LLM efficiency methods surfaced a number of
defects in the **authors' released artifacts** — not in our harness. This page
records them precisely, with the evidence, so they are citable and so future
replicators do not re-derive them.

Two categories:

* **§1 Blocking** — the method cannot be run from its published artifacts at all.
* **§2 Non-blocking** — real defects we worked around; the method runs afterwards.

Everything here was confirmed by reading the released source and by controlled
experiment, not inferred from a stack trace alone.

---

## 1. Blocking: DyTo is not reproducible from published artifacts

DyTo (ICCV 2025, LLaVA-NeXT Vicuna-7B) contains **two independent defects**, either
of which alone prevents it from running.

### 1.1 The code requires an argument its own pinned dependency does not have

`dyto/llava/model/llava_arch.py:192`:

```python
c, num_clust, _ = FINCH(image, verbose=False, tw_finch=tw_finch)   # tw_finch=True at :233
```

| Evidence | Finding |
|---|---|
| DyTo's declared dependency (`pyproject.toml`, `dyto.egg-info/PKG-INFO`) | **`finch-clust==0.2.0`** — an exact PyPI version, *not* a fork or git URL |
| Version installed in our env | **0.2.0** — the pin is already satisfied |
| Pristine 0.2.0 downloaded from PyPI and unpacked | signature is `FINCH(data, initial_rank, req_clust, distance, ensure_early_exit, verbose, use_ann_above_samples)` — **`tw_finch` occurs 0 times** |
| Our installed copy vs pristine | identical — no local corruption |
| Upstream `ssarfraz/FINCH-Clustering` | TW-FINCH is a **separate implementation**, not a parameter of `FINCH()` |

The authors evidently developed against a local FINCH modification that was never
published or referenced. **This is not a version mismatch a replicator can
correct** — the pinned version *is* the installed version.

### 1.2 `finch_cluster()` has no return statement

Found by building an opt-in shim that drops the unsupported kwarg, so the rest of
DyTo could run as a clearly-labelled variant. The shim works (it rebinds `FINCH`
in 3 already-imported modules and the `tw_finch` error clears), and execution then
reaches:

```
llava_arch.py:234   if clus_image_features.shape[0] > 25:
AttributeError: 'NoneType' object has no attribute 'shape'
```

`finch_cluster()` (`llava_arch.py:188–218`) computes `classification`, builds
`selected_indexs` — and then the function body simply ends; the next `def
encode_images` begins. **It contains zero `return` statements** and therefore
returns `None` unconditionally, which its caller at line 233 immediately
dereferences.

Verified independently: `grep` over lines 188–218 finds one `def` and no `return`.
Standard FINCH runs fine on this input (`c.shape=(32,2)`, `num_clust=[8,3]`), so
this is **not** an artefact of our shim — the released function is incomplete.

### 1.3 Why we did not "fix" it

Completing `finch_cluster()` means inventing the missing logic: guessing which
tokens the authors intended to select and how to assemble them. That reconstruction
would be *our* algorithm published under DyTo's name — the precise failure mode this
project exists to prevent (see [METHODOLOGY.md](METHODOLOGY.md)). Even a
labelled-variant run is impossible, because the variant hits defect 1.2.

**Status: not reproducible from published artifacts.** Everything upstream of the
clustering is fixed and working — the Vicuna backbone loads, prompts build, video
decodes, patch-merging is correct, and execution reaches DyTo's own aggregation
stage. The blocker is entirely inside DyTo's released `finch_cluster`.

*(Four earlier DyTo bugs were ours or environmental and are fixed: a swallowed
import error, a non-existent conv template, a 100-frame OOM, and an anyres/tensor
mismatch. See [PORT_FEASIBILITY.md](../stage3-qwen3-vl/PORT_FEASIBILITY.md).)*

---

## 2. Non-blocking defects and incompatibilities

These were real obstacles in released code or configuration. Each is fixed; the
method runs and is gate-verified afterwards.

| # | Method / component | Defect | Fix |
|---|---|---|---|
| 1 | **DyTo** `dyto/llava/model/__init__.py` | Imports wrapped in `try/except: pass`, silently swallowing a genuine `ModuleNotFoundError` from an **absolute** `from llava.constants import ...` (the package vendors LLaVA as `dyto.llava`). Surfaces as the misleading `cannot import name 'LlavaLlamaForCausalLM'` — cost 5 failed jobs before diagnosis. | Register `dyto.llava` under the name `llava` before any DyTo import |
| 2 | **DyTo** + `llava-v1.6-vicuna-7b` config | Weights ship `mm_patch_merge_type="spatial_unpad"` / `image_aspect_ratio="anyres"`, which builds a **list** of per-patch features, but DyTo's own `temporal_aggregation` does `T, N, D = image_features.shape` and needs a **3-D tensor**. The two released paths are mutually incompatible for video. | Force `flat` / `square` for the video path |
| 3 | **MDP3** `vlmeval` package | Importing the frame selector pulls the whole package, which does `from transformers import AutoModelForVision2Seq` — **removed in transformers 5.x**, which Qwen3-VL requires. The two cannot coexist. | Load `mdp3_frame_selector.py` directly by file path, bypassing the package |
| 4 | **PruneVID** `modeling_pllava.py` | Uses relative imports, so loading the file standalone raises *"attempted relative import with no known parent package"* | Import with proper package context |
| 5 | **LLaVA-OV** (all methods) | `attn_implementation="eager"` produces **100% empty generations** — a silent failure that looks like a clean, completed run | Load with `sdpa`; recompute attention per-layer where a method needs it |
| 6 | **DyCoke** builder | Defaults to `flash_attention_2`, which is not installed → `ImportError` | Pass `attn_implementation="sdpa"` explicitly |
| 7 | **Qwen3-VL** `Qwen3VLVideoProcessor` | `do_sample_frames=True, fps=2` means it **re-samples frames and ignores `--num_frames`**, warning only *"Defaulting to fps=24"*. Runs silently use the wrong frame count. | Pass `do_sample_frames=False`; assert the delivered count |
| 8 | **VisionZip** (our eval) | `output_ids[:, input_ids.shape[1]:]` — LLaVA-1.5's `generate()` returns **only new tokens**, so this slice discarded every response (100% empty) | Decode `output_ids` directly |

---

## 3. Architectural incompatibilities (not defects)

Recorded to distinguish "the code is broken" from "the method does not apply".

| Method | On Qwen3-VL | Why |
|---|---|---|
| **VisionZip** (dominant half) | ❌ | Selects tokens by **CLS-attention**; Qwen3-VL's vision tower has **no CLS token**. The *contextual* half is portable and runs as a labelled partial |
| **STTM** | ❌ | Ships a wholesale `Qwen2Model_forward` replacement with no separable merge routine. Porting = rewriting Qwen3's decoder forward |
| **PruneVID / DyTo** | ❌ | Bound to PLLaVA / Vicuna backbones respectively; evaluated there instead |

---

## 4. What this suggests about the field

Of eleven methods, **one is not reproducible from its published artifacts at all**,
and most required non-trivial fixes to released code or configuration before they
would run. Several defects are of a kind that unit tests or a single end-to-end
CI run would catch — a missing `return`, a dependency pin that contradicts the
code, a `try/except: pass` that hides a real import error.

The recurring hazard is **silent failure**: eager attention producing empty output,
a processor quietly re-sampling frames, an exception swallowed by a bare `except`.
Each looks like a clean run and reports a plausible number. That is why this project
requires every recorded result to pass a divergence gate — see
[METHODOLOGY.md](METHODOLOGY.md) and
[DETERMINISM_AND_VALIDITY.md](DETERMINISM_AND_VALIDITY.md).

**Nothing here is a criticism of the underlying research.** These are engineering
defects in release artifacts, of the sort that surface only when someone tries to
run the code on a different machine — which is precisely what replication is for.

# Qwen3-VL Port Feasibility — 11 methods

Verified against the real architecture (`transformers.models.qwen3_vl.modeling_qwen3_vl`)
and each method's actual source, not assumed. Facts that govern every port:

* `Qwen3VLTextModel` — 36 decoder layers.
* `Qwen3VLTextAttention.forward` returns `(attn_output, attn_weights)`, but
  **`attn_weights` is `None` under sdpa** and loading the model eager breaks
  generation (100% empty output). Recompute the row you need from q/k instead.
* `Qwen3VLTextDecoderLayer.forward` returns a **bare tensor**, not a tuple.
* `hidden_states` arrives as a **kwarg**; the positional args tuple is empty.
* **`visual_pos_masks`** gives exact visual-token positions.
* Vision tower: **no CLS token**; attention returns only `attn_output`; but the
  block has a **fused `qkv`** projection, so k-vectors are recoverable.

---

## ✅ Ported and gate-verified (7)

| Method | Mechanism as implemented | Attention needed? |
|---|---|---|
| **FastV** | Attention-rerank at AGG_LAYER; keep top 15% | yes (recomputed from q/k) |
| **DyCoke** | Temporal merge (k) + attention prune at layer l (p) | yes (recomputed) |
| **HoliTom** | Outer segment retain + inner attention merge at layer k | yes (recomputed) |
| **FlashVID** | Pre-LLM segment merge, α·saliency + (1−α)·distinctiveness | no |
| **AIM** | Bipartite soft-matching merge + PageRank prune | no |
| **MDP3** | Conditional-DPP frame selection before the model | no |
| **VideoITG** | Grounded frame selection (stage-1 indices reused) | no |

---

## Re-assessment of the four "unrunnable" methods (2026-07-24)

An earlier pass marked these ❌ wholesale. Re-reading each method's **source**
shows that was too coarse — two are partly portable, two genuinely are not.

### 🟡 VisionZip — ~50% portable (was ❌)

VisionZip is **two independent halves** (`visionzip/clip_encoder.py`):

1. **Dominant tokens** — `cls_attention = attn_weights[:, :, 0, 1:]`, top-k.
   **Blocked**: needs a CLS token, which Qwen3-VL's vision tower does not have.
2. **Contextual tokens** — merges the *remaining* tokens by cosine similarity of
   `metric`, where `metric` is the **key vectors of encoder layer −2**. Uses no
   CLS whatsoever.

The `metric` half is fully reproducible: Qwen3-VL's `Qwen3VLVisionAttention` has a
fused `qkv`, and k is recoverable —
`qkv(hs).reshape(s,3,num_heads,-1).permute(1,0,2,3)[1]` → `(s, 1152)`. **Verified
working on Carya.**

**Honest options:** (a) implement contextual-only and label it
**"VisionZip (contextual-only)"** — a documented partial, not the published method;
(b) substitute a CLS proxy (e.g. mean attention received, or patch-merger score)
for the dominant half and label it **"VisionZip-variant"**. Either is legitimate
*if labelled*; neither may be reported as "VisionZip".

### 🟡 PruneVID — core is separable (was ❌)

Its VTP primitives in `models/pllava/modeling_pllava.py` — `cluster_dpc_knn`,
`refine_clusters`, `compute_cluster_vectors`, `spatial_merge_tokens` — are
**standalone functions over key-vectors**, not PLLaVA-coupled. They could drive a
Qwen3-VL port using `visual_pos_masks` + recomputed k.

**Precondition:** PruneVID's *LLaVA-OV* port is still inert (0/8052 divergence). It
would be wrong to build a Qwen3-VL port on a mechanism we have not yet made work
once. Fix LLaVA-OV first, then this becomes a normal port.

### 🟡 STTM — portable after all (verdict CORRECTED 2026-07-25)

**The earlier "not portable" verdict was wrong.** It was reached by reading only
`token_merging_monkey_patch/quadtree_attn_monkey_patch.py` (the LLaVA/Qwen2 one)
and concluding the merge was welded into a 180-line forward replacement. Two facts
were missed:

1. **The authors already ship a Qwen2-**VL** port** —
   `token_merging_qwen2vl_monkey_patch/quadtree_attn_monkey_patch.py`. Qwen2-VL is
   far closer to Qwen3-VL than the LLaVA/Qwen2 path, so this is the template.
2. **The merge IS a separable function.** Both patches delegate to
   `get_quadtree_features()` in `token_merging_utils/quadtree_interface.py`, which
   contains **zero** references to llava / transformers / Qwen (verified by grep).
   It is pure tensor math: features in, merged features + coordinates out.

What the forward actually does around that call is bookkeeping, ~25 lines:
slice sys/visual/inst spans, `rearrange` the visual tokens to `(T C H W)`, call
`get_quadtree_features`, concatenate the merged tokens back, and re-derive
`position_ids` / `position_embeddings` / `cache_position` for the new (shorter)
sequence.

**Contract to satisfy on Qwen3-VL:** a `(T, C, H, W)` visual grid plus scalar
`T`, `H`, `W`. All are available — `visual_pos_masks` gives exact visual token
positions, and `video_grid_thw` (43 occurrences in `modeling_qwen3_vl`) with
`spatial_merge_size` (20) gives the grid dimensions.

**One genuine difference from the other seven ports:** every gate-verified
Qwen3-VL port so far *masks* tokens (sequence length unchanged). STTM physically
*shortens* the sequence, so position_ids and cache_position must be rebuilt — the
Qwen2-VL patch shows exactly how, but Qwen3-VL's mRoPE and `deepstack_visual_embeds`
mean it is not a copy-paste. This is real work, but it is porting, not inventing.

**Grid arithmetic, measured (not assumed).** The quadtree needs a regular
`[T, C, H, W]` layout, so the token count must factor cleanly:

* Qwen3-VL emits **11,664 visual tokens** at `--num_frames 32` (confirmed in the
  ACTIVE logs of all six token-level ports).
* `11664 / 32 = 364.5` — **not an integer**. The naive "32 frames = 32 grid rows"
  assumption is wrong.
* `vision_config.temporal_patch_size = 2`, so 32 frames become **T = 16** temporal
  positions of `11664/16 = 729 = 27×27` tokens. `patch_size=16`,
  `spatial_merge_size=2`.
* **27 is odd**, and the quadtree halves sides per level. Not a blocker: the
  authors' `avgpool_to_even_side_feature` already handles odd sides explicitly
  (`math.ceil(h/2)` with separate odd-height / odd-width branches).

So the grid is `T=16, H=W=27`, and STTM's builder accepts it as-is.

**Remaining work, in order:** (1) recover `T,H,W` from `video_grid_thw` at runtime
rather than hardcoding; (2) slice sys/visual/inst spans via `visual_pos_masks`;
(3) `rearrange` to `[T, C, H, W]` and call `get_quadtree_features`; (4) splice the
merged tokens back and rebuild `position_ids` / `position_embeddings` /
`cache_position`; (5) handle `deepstack_visual_embeds`, which the Qwen2-VL patch
has no equivalent for — this is the one genuinely novel piece and the most likely
source of a silent no-op, so it must be gate-checked for divergence, not just for
an ACTIVE log.

### ❌ DyTo — genuinely not portable

The mechanism is selected inside **`dyto/llava/model/llava_arch.py:231`**
(`temporal_aggregation == "spatial_tome_finch_dynamic_all_frms"`), i.e. it lives in
LLaVA's multimodal pipeline and depends on LLaVA's frame/patch handling, and the
package vendors its own `llava` plus a Vicuna-only builder. Qwen3-VL shares none
of that. DyTo remains valid **on its own backbone** (Vicuna) — which is where we
run it.

---

## Summary

| Verdict | Methods |
|---|---|
| ✅ Ported + verified | FastV, DyCoke, HoliTom, FlashVID, AIM, MDP3, VideoITG (7) |
| 🟡 Partially portable, needs a labelled variant | VisionZip (contextual half) |
| 🟡 Blocked on an upstream fix, not on architecture | PruneVID (fix LLaVA-OV port first) |
| 🟡 Portable, not yet built | STTM (verdict corrected 2026-07-25) |
| ❌ Not portable | DyTo (bound to its Vicuna backbone) |

**Principle unchanged:** a cell that cannot be done authentically is recorded as a
hole with the blocker stated — never filled with a baseline wearing the method's
name (the FastV-stub / PruneVID-OV failure this project already had to correct).
A *partial* implementation is acceptable **only** when labelled as such.

---

## DyTo (LLaVA-NeXT Vicuna) — blocked on an upstream dependency, 2026-07-24

DyTo is its own backbone track (not a Qwen3-VL port), but it is the last unresolved
cell, so the diagnosis is recorded here. Four real bugs were fixed in sequence, each
revealed only after the previous one cleared:

| # | Symptom | Root cause | Status |
|---|---|---|---|
| 1 | `ImportError: cannot import name 'LlavaLlamaForCausalLM'` | `dyto/llava/model/__init__.py` wraps its imports in `try/except: pass`, swallowing a real `ModuleNotFoundError` from an **absolute** `from llava.constants import ...` | ✅ fixed (alias `dyto.llava` as `llava`) |
| 2 | `KeyError: 'image_seq_v3'` per sample | that conv template does not exist; the Vicuna backbone needs `vicuna_v1` | ✅ fixed |
| 3 | CUDA OOM at 100 frames | needs ~5 GiB more than the 44 GiB card provides | ✅ fixed (32 frames; FINCH selects ~25 regardless) |
| 4 | `'list' object has no attribute 'shape'` at `llava_arch.py:325` | the weights ship `mm_patch_merge_type="spatial_unpad"` / `image_aspect_ratio="anyres"`, which builds a **list** of per-patch features, but DyTo's own `temporal_aggregation` does `T, N, D = image_features.shape` and needs a **3-D tensor**. The two paths are mutually incompatible for video. | ✅ fixed (force `flat`/`square`) |
| 5 | `FINCH() got an unexpected keyword argument 'tw_finch'` | **BLOCKED** — see below | ❌ open |

### The remaining blocker — RESOLVED as an upstream inconsistency

`dyto/llava/model/llava_arch.py:192` calls

```python
c, num_clust, _ = FINCH(image, verbose=False, tw_finch=tw_finch)   # tw_finch=True at :233
```

We chased the exact dependency rather than assume. Findings:

| Evidence | Result |
|---|---|
| DyTo's own pin (`pyproject.toml`, `PKG-INFO`) | **`finch-clust==0.2.0`** — an exact version, *not* a fork or git URL |
| Version installed on Carya | **0.2.0** — the pin is already satisfied |
| Pristine 0.2.0 downloaded from PyPI and unpacked | `def FINCH(data, initial_rank=None, req_clust=None, distance=..., ensure_early_exit=..., verbose=..., use_ann_above_samples=...)` — **`tw_finch` appears 0 times** |
| Our installed copy vs pristine | identical (both 0 occurrences) — no local corruption |
| Upstream `ssarfraz/FINCH-Clustering` | keeps **TW-FINCH as a separate implementation**, not a parameter of `FINCH()` |

**Conclusion: DyTo's published artifacts are internally inconsistent.** Its code
requires `FINCH(..., tw_finch=...)`, but the dependency it pins has never exposed
that argument in any release. The authors must have developed against a local
FINCH modification that was neither published nor referenced. No amount of
environment work on our side can resolve this — it is not a version mismatch we
can correct, because the pinned version *is* installed.

**Therefore DyTo is recorded as: not reproducible from published artifacts.**

### Second, independent defect found while attempting the labelled variant

We built an opt-in shim (`$DYTO_FINCH_SHIM=1`) that drops the unsupported kwarg so
the rest of DyTo could run as a clearly-labelled variant. The shim works — it
rebinds `FINCH` in **3 already-imported modules** and the `tw_finch` error clears.
Execution then advances to the next failure:

```
llava_arch.py:234  if clus_image_features.shape[0] > 25:
AttributeError: 'NoneType' object has no attribute 'shape'
```

Cause: **`finch_cluster()` (llava_arch.py:188–218) contains no `return` statement
at all.** It computes `classification`, builds `selected_indexs`, and then the
function body simply ends — the next `def encode_images` begins. It therefore
returns `None` unconditionally, and its caller at line 233 immediately dereferences
`.shape` on that `None`.

Verified: `grep` over lines 188–218 finds exactly one `def` and **zero** `return`
statements. Standard FINCH itself works fine on this input (`c.shape=(32,2)`,
`num_clust=[8,3]`), so this is not a consequence of our shim — the published
function is incomplete.

### Final verdict on DyTo

Two independent defects in the published artifacts:

1. Code calls `FINCH(..., tw_finch=...)`; the pinned `finch-clust==0.2.0` has never
   exposed that argument.
2. `finch_cluster()` has no return statement and always yields `None`.

Neither is fixable without **inventing** the missing logic — i.e. guessing what
tokens the authors intended to select and how. Any such reconstruction would be
our algorithm, not DyTo's, and reporting it under DyTo's name is exactly the
failure mode this project refuses. **DyTo is not reproducible from its published
artifacts, and no labelled-variant run is possible either** — the variant attempt
was made in good faith and is documented here as evidence, not abandoned untried.

Everything upstream of the clustering is fixed and working: the Vicuna backbone
loads, prompts build, video decodes, patch-merge is correct, and execution reaches
DyTo's own aggregation stage. The blocker is entirely inside DyTo's published
`finch_cluster`.

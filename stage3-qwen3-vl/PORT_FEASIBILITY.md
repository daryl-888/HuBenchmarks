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

### ❌ STTM — genuinely not portable

`replace_qwen2_with_quadtree_attn` does not expose a reusable merge function: the
package ships a **wholesale `Qwen2Model_forward` replacement** (~180 lines) plus
class-attribute assignment onto `transformers.models.qwen2.modeling_qwen2`. There
is no separable quadtree routine to lift. Porting = rewriting Qwen3's decoder
forward around a quadtree — a research contribution, not a port.

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
| ❌ Not portable | STTM, DyTo (2) |

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

Fabricating a `tw_finch` shim (e.g. routing to standard FINCH, or hand-porting
TW-FINCH) would silently substitute a different clustering algorithm than the paper
specifies. That is precisely the "variant reported as the method" failure this
project exists to prevent. If a number is wanted anyway, the only honest form is a
clearly-labelled variant — **"DyTo (standard FINCH, not TW-FINCH)"** — never "DyTo".

Progress note: bugs 1–4 are genuinely fixed. DyTo now loads its Vicuna backbone,
builds prompts, processes video, and reaches its FINCH clustering stage. Only the
unresolvable dependency remains.

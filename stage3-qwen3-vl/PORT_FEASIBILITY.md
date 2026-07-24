# Qwen3-VL Port Feasibility — 11 methods

Verified against the real architecture (`transformers.models.qwen3_vl.modeling_qwen3_vl`),
not assumed. Facts that govern every port:

* `Qwen3VLTextModel` — 36 decoder layers.
* `Qwen3VLTextAttention.forward` returns **`(attn_output, attn_weights)`** → hook the
  **attention module** when you need attention scores.
* `Qwen3VLTextDecoderLayer.forward` returns a **bare tensor, not a tuple** → a
  layer-level hook reading `out[1]` silently gets `None` and no-ops.
* **`visual_pos_masks`** (bool `[B, S]`) gives exact visual-token positions — much
  better than LLaVA-OV, where the image span had to be inferred from a 14-token prefix.
* The vision tower (`Qwen3VLVisionModel`) has **NO CLS token**.

---

## ✅ Ported (7)

| Method | Mechanism as implemented | Needs attention? |
|---|---|---|
| **FastV** | Attention-rerank at AGG_LAYER; keep top 15% of visual tokens | yes (eager) |
| **DyCoke** | Stage 1 temporal merge by embedding similarity (k) + Stage 2 attention prune at layer l (p) | yes (eager) |
| **HoliTom** | Outer temporal-segment retain (T, RETAIN_RATIO) + inner attention merge at layer k (r) | yes (eager) |
| **FlashVID** | Pre-LLM temporal-segment merge; score = α·saliency + (1−α)·distinctiveness | **no** (sdpa) |
| **AIM** | Bipartite soft-matching merge (4 steps) + PageRank centrality prune | **no** (sdpa) |
| **MDP3** | Conditional-DPP frame selection **before** the model (pool 32 → select 8) | **no** |
| **VideoITG** | Grounded frame selection; stage-1 `frame_indices` reused verbatim | **no** |

MDP3 and VideoITG are the cleanest ports: both act on *frames*, before the model,
so they are genuinely architecture-independent. FlashVID and AIM need only input
embeddings, so they avoid the eager-attention cost.

---

## ❌ Not portable as-published (4)

| Method | Blocker | Honest options |
|---|---|---|
| **VisionZip** | Selects "dominant" tokens by **CLS-attention** score inside the vision tower. **Qwen3-VL's vision tower has no CLS token** — the signal the method is defined on does not exist. | Either document as incompatible, or implement a *substitute* saliency (e.g. mean-attention or patch-merger score). That would no longer be VisionZip-as-published, so it must be labelled a variant, not the method. |
| **STTM** | Patches **Qwen2** attention specifically (`replace_qwen2_with_quadtree_attn`, importing `transformers.models.qwen2.modeling_qwen2`). Qwen3 attention is a different module. | A Qwen3 quadtree re-implementation is a research project in itself, not a port. |
| **PruneVID** | Targets PLLaVA/LLaVA VTP internals. Its LLaVA-OV port already proved inert (0/8052 predictions differed from baseline). | Do not attempt on Qwen3-VL until the LLaVA-OV port itself works. |
| **DyTo** | Hard-bound to LLaVA-NeXT **Vicuna**; ships its own vendored `llava` package and a Vicuna-only builder. | Out of scope for Qwen3-VL. |

**Principle:** a cell that cannot be done authentically is recorded as a hole with
the blocker stated. It is never filled with a baseline wearing the method's name —
that is exactly the failure mode (FastV stub, PruneVID-OV) this project already had
to dig itself out of.

---

## Verification requirement

Every port logs `<method>(Qwen3-VL) ACTIVE: ...` and the attention-dependent ones
warn loudly when attention weights are unavailable. No port may be recorded until:

```bash
python3 scripts/check_run.py <run> --expect-method <m> --smoke \
    --vs-baseline /project/rhu/dpalfaro/results/qwen3vl_baseline_run1
```

0 differing predictions vs the Qwen3-VL baseline ⇒ silent no-op ⇒ not a result.

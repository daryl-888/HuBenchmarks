# Backbone: Qwen3-VL-8B

**Weights** `Qwen/Qwen3-VL-8B-Instruct` (`$W_QWEN3VL`) ·
**Arch** `Qwen3VLForConditionalGeneration` (native HuggingFace, **not** a LLaVA
fork) · **LLM** Qwen3, 36 text-decoder layers · **Era** late-2025 ·
**Env** `qwen3vl` only (torch 2.6.0+cu124, transformers 5.14.1)

## Baseline (32 frames, no method) — the headline

**62.52%** (2512/4018) — **+9.86 over LLaVA-OV-7B**, and higher in every category.

| Backbone | Overall | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-VL-8B** | **62.52%** | 46.1 | 63.1 | 65.0 | 67.3 | 79.0 | 33.8 |
| LLaVA-OV-7B | 52.66% | 40.5 | 45.2 | 55.5 | 57.0 | 71.2 | 23.8 |
| **Δ** | **+9.86** | +5.6 | **+17.9** | +9.5 | +10.3 | +7.8 | +10.0 |

Largest gain: **Camera Motion (+17.9)**. Even the hardest category for both
(Repetition Count) rises 23.8 → 33.8.

## Why Stage 3 needed real re-implementation

Qwen3-VL is not a LLaVA fork, so **no method's reference code transfers** — the
algorithm ports, the integration is new. The original 8 Stage-3 scripts were all
the *same baseline* applying no method (now archived; a loud-fail guard remains).
Each real port hooks `Qwen3VLTextModel` directly.

Two architecture facts govern every port:
- `Qwen3VLTextAttention.forward` returns `(attn_output, attn_weights)` → hook the
  **attention module** when attention is needed.
- `Qwen3VLTextDecoderLayer.forward` returns a **bare tensor** → a layer-level hook
  reading `out[1]` silently no-ops.
- **`visual_pos_masks`** gives exact visual-token positions (cleaner than
  LLaVA-OV, where the span is inferred from a fixed prefix).

## Method port status

| Method | Status | Notes |
|---|:---:|---|
| Baseline | ✅ 62.52% | verified full run |
| MDP3, VideoITG | 🔄 ported | frame-selection, model-agnostic — no attention needed |
| FlashVID, AIM | 🔄 ported | embedding-based scoring — run under sdpa |
| FastV, DyCoke, HoliTom | 🔄 ported | attention-based; first smoke ran as baseline (eager broke generation), reworked to sdpa, verifying |
| VisionZip | ❌ incompatible | needs a **CLS token Qwen3-VL's vision tower lacks** |
| STTM | ❌ incompatible | patches **Qwen2** attention specifically |
| PruneVID, DyTo | ❌ incompatible | bound to PLLaVA / Vicuna backbones |

See [`stage3-qwen3-vl/PORT_FEASIBILITY.md`](../../stage3-qwen3-vl/PORT_FEASIBILITY.md)
for the full per-method feasibility analysis.

## Backbone-specific pitfall

`Qwen3VLVideoProcessor` has `do_sample_frames=True, fps=2` — it **re-samples your
frames and ignores `--num_frames`** (warns "Defaulting to fps=24"). Pass
`do_sample_frames=False` to the processor. Verify with the `FRAMES: requested=32
given=32` log line.

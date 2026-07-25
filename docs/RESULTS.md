# Results

All numbers on **MotionBench** (4,018 scoreable), 32 frames, greedy decoding.
✅ = verified engaged (passed the divergence gate) · 🔄 = running/unverified ·
❌ = not a result (no-op / crash / incompatible). See
[METHODOLOGY.md](METHODOLOGY.md) for what "verified" means.

## Baselines

| Backbone | Overall | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-VL-8B** | **62.52%** | 46.1 | 63.1 | 65.0 | 67.3 | 79.0 | 33.8 |
| LLaVA-OV-7B | 52.66% | 40.5 | 45.2 | 55.5 | 57.0 | 71.2 | 23.8 |

**The backbone dominates the method** — a ~10-point gap no efficiency method closes.

## LLaVA-OV-7B methods ✅ all gated

| # | Method | Venue | Year | Overall | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|:-:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [DyCoke](methods/dycoke.md) | arXiv | 2024 | **53.36%** | 38.9 | 48.8 | 55.7 | 58.2 | 70.9 | 25.2 |
| 2 | [FlashVID](methods/flashvid.md) | ICLR (Oral) | 2026 | **53.31%** | 39.9 | 45.7 | 53.8 | 58.0 | 71.7 | 28.2 |
| 3 | [HoliTom](methods/holitom.md) | — | 2025 | **53.14%** | 40.8 | 49.6 | 52.6 | 57.0 | 71.4 | 27.2 |
| 4 | [MDP3](methods/mdp3.md) | ICCV | 2025 | **53.06%** | 40.5 | 49.6 | 53.5 | 56.8 | 71.6 | 26.5 |
| 5 | [AIM](methods/aim.md) | ICCV | 2025 | **52.86%** | 41.4 | 48.1 | 54.2 | 57.0 | 71.9 | 22.2 |
| 5 | [VideoITG](methods/videoitg.md) | — | 2025 | **52.86%** | 40.1 | 47.0 | 53.7 | 57.6 | 70.1 | 26.8 |
| — | *baseline* | — | — | *52.66%* | *40.5* | *45.2* | *55.5* | *57.0* | *71.2* | *23.8* |
| 7 | [STTM](methods/sttm.md) | — | 2025 | **51.72%** | 39.9 | 48.1 | 53.8 | 53.5 | 70.6 | 28.8 |
| 8 | [VisionZip](methods/visionzip.md) | — | 2024 | **40.09%** | 33.7 | 33.0 | 37.2 | 41.1 | 57.4 | 25.8 |
| 9 | [FastV](methods/fastv.md) | arXiv | 2024 | **36.78%** | 32.8 | 31.9 | 34.1 | 35.7 | 53.8 | 25.2 |

**Parameters used** (standardized to 32 frames; 15% retention where the method exposes one):
DyCoke `l=3,p=0.7,k=0.7` · FlashVID `retention=0.15,α=0.7` · HoliTom `RETAIN=0.15,T=0.80,k=18,r=0.5` ·
MDP3 `pool=32,select=8` · AIM 4-step bipartite merge + PageRank · VideoITG 512 sampled/32 selected ·
STTM `layer=2,thresh=0.85` · VisionZip `dominant=54,contextual=10` · FastV `k=2,r=0.85`
(**paper default is r=0.5** — our 15% is far more aggressive, which explains the drop).

Per-model mechanism, full parameters and per-method notes:
[master-results.md §3b](../memory-bank/claude/master-results.md).

Six of seven methods land **within measurement resolution** (±1.54 pts) of the backbone — i.e. accuracy-neutral on MotionBench at a 15% budget — while FastV and VisionZip degrade significantly. Tested properly (McNemar on paired predictions) the best of them, DyCoke, gives **χ² = 2.34** vs the 3.84 needed for p<0.05 — i.e. **not significant**. Treat the ordering below as unranked: on LLaVA-OV these methods are indistinguishable from the backbone and from each other. See [DETERMINISM_AND_VALIDITY.md](DETERMINISM_AND_VALIDITY.md). FastV (and
VisionZip, below) drop hard because they were run at the standardized 15% retention,
far more aggressive than their paper defaults; the degradation is real (thousands of
predictions differ from baseline), not a bug.

## Qwen3-VL-8B methods

**All 8 methods are fully gated** (2026-07-25). Each ran 8,052/8,052 samples at 32
frames (~9–10h each — Qwen3-VL carries 11,664 visual tokens per sample), printed its
own `ACTIVE` log, and diverged from the baseline. Zero tracebacks; the uniform 8
empty predictions per run are the known bad-NFS videos, not a method defect.

| # | Method | Overall | Δ vs base | Differ | χ² | Significant? |
|:-:|---|:---:|:---:|:---:|:---:|:---:|
| — | **Baseline** | **62.52%** | — | 0 (ref) | — | — |
| 1 | DyCoke ✅ | **61.85%** | −0.67 | 1048 | 3.5 | **no** |
| 2 | HoliTom ✅ | **60.33%** | −2.19 | 1657 | 21.6 | yes |
| 3 | MDP3 ✅ | **59.66%** | −2.86 | 1626 | 30.0 | yes |
| 4 | FastV ✅ | **59.01%** | −3.51 | 2029 | 44.6 | yes |
| 5 | VisionZip (contextual-only) ✅ | **58.81%** | −3.71 | 2035 | 41.3 | yes |
| 6 | FlashVID ✅ | **56.65%** | −5.87 | 2625 | 93.3 | yes |
| 7 | VideoITG ✅ | **56.35%** | −6.17 | 2522 | 92.4 | yes |
| 8 | AIM ✅ | **55.97%** | −6.55 | 2861 | 105.4 | yes |
| — | PruneVID | 🟡 not built | — | — | — | VTP core is separable; its LLaVA-OV port is now fixed, so this is unblocked |
| — | STTM, DyTo | ❌ | — | — | — | genuinely not portable ([why](../stage3-qwen3-vl/PORT_FEASIBILITY.md)) |

Subcategory breakdown (% correct within category):

| Method | Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| *Baseline* | *46.1* | *63.1* | *65.0* | *67.3* | *79.0* | *33.8* |
| DyCoke | 45.1 | 62.6 | 63.9 | 66.4 | 78.4 | 34.5 |
| HoliTom | 44.7 | 61.3 | 61.0 | 66.4 | 76.2 | 29.0 |
| MDP3 | 44.9 | 64.7 | 62.8 | 64.0 | 77.4 | 23.0 |
| FastV | 46.2 | 61.0 | 61.9 | 63.5 | 72.2 | 30.5 |
| VisionZip | 43.2 | 57.4 | 61.9 | 64.2 | 74.3 | 29.5 |
| FlashVID | 43.4 | 57.9 | 57.1 | 61.2 | 73.0 | 27.0 |
| VideoITG | 45.3 | 53.5 | 59.3 | 60.2 | 75.4 | 22.2 |
| AIM | 45.9 | 56.6 | 59.5 | 58.3 | 72.2 | 27.0 |

> ### Every method loses, and loss tracks how hard it prunes
> **7 of 8 lose significantly.** Only DyCoke is indistinguishable from the backbone —
> and it retains **49%** of tokens where the rest cut to 15%. AIM and FlashVID prune
> hardest and lose most. Repetition Count is the weakest category throughout (23–34%).
>
> **This is the opposite of Stage 1.** On LLaVA-OV no method's change was
> significant; on Qwen3-VL, at identical settings, all 8 lose. A stronger backbone
> extracts more from the full token set, so discarding tokens costs more. Method
> rankings measured on one backbone do not transfer to another.

> **Getting these 7 to engage took untangling a 5-bug chain** where each bug hid the
> next — `attention_mask` is `None` under sdpa, bf16 overflow from an fp32 mask,
> `hidden_states` passed as a kwarg, an undefined variable, and a transformers-5.x
> import conflict. Two ports printed `ACTIVE` while producing byte-identical output
> to the baseline; only the divergence gate caught it. See
> [METHODOLOGY.md](METHODOLOGY.md#hardening-divergence-checking-is-mandatory-2026-07-24).

## Other backbones (each method's native model)

| Method | Backbone | Overall | Status |
|---|---|:---:|---|
| [PruneVID](methods/prunevid.md) | PLLaVA-7B | **44.13%** | ✅ (its published backbone) |
| PruneVID (LLaVA-OV port) | LLaVA-OV-7B | 🔄 pending | ✅ **now engages** — 4/8 divergence after switching to `PrunableDynamicCache.kv_cache`; was inert |
| [VisionZip](methods/visionzip.md) | LLaVA-1.5-7B | **39.97%** | 🟡 |
| STTM-LLaVAVid | LLaVA-Video-7B | **53.33%** | 🟡 |
| [DyTo](methods/dyto.md) | Vicuna-7B | ❌ | **Not reproducible from published artifacts** — two defects in released code ([evidence](UPSTREAM_DEFECTS.md#1-blocking-dyto-is-not-reproducible-from-published-artifacts)) |
| iMove, TrajViT | — | — | ❌ no public code |

## Known-invalid numbers (do not cite)

| Label | Number | Why |
|---|:---:|---|
| "FastV" (old) | 52.66% | stub, `enabled:false` — bare backbone |
| "PruneVID (OV port)" | 52.66% | 0/8052 diverge — VTP never fired |
| "VisionZip" (old) | 0.00% ×3 | output-slicing bug (fixed) |
| "DyTo" (old) | 5.25% | emitted captions not letters — below random |

Full working record: [`memory-bank/claude/master-results.md`](../memory-bank/claude/master-results.md).

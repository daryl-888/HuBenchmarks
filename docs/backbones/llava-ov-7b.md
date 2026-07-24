# Backbone: LLaVA-OV-7B

**Weights** `lmms-lab/llava-onevision-qwen2-7b-ov` (`$W_LLAVA_OV`) ·
**LLM** Qwen2-7B (`LlavaQwenForCausalLM`, hidden_size 3584) ·
**Vision** SigLIP-so400m-patch14-384 · **Era** mid-2024 · **Conv template** `qwen_2`

> **Note on "Qwen1.5 vs Qwen2".** Earlier project notes tracked two backbones,
> `ovqwen15` and `ovqwen2`. They are the **same model** — `llava-ov-7b` uses Qwen2
> internally. `qwen_1_5` vs `qwen_2` is only a **prompt-template** difference, not
> different weights. A duplicate `llava-ov-7b-qwen2` weights dir was a byte-for-byte
> copy and was deleted. Identical results across the two are expected.

## Baseline (32 frames, no method)

**52.66%** (2116/4018)

| Act.Order | Cam.Motion | Loc.Motion | Mot.Rec. | Mot.Obj. | Rep.Count |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 40.5 | 45.2 | 55.5 | 57.0 | 71.2 | 23.8 |

## Methods on this backbone (all ✅ gated)

| Method | Overall | vs baseline | Page |
|---|:---:|:---:|---|
| DyCoke | **53.36%** | +0.70 | [dycoke](../methods/dycoke.md) |
| FlashVID | **53.31%** | +0.65 | [flashvid](../methods/flashvid.md) |
| HoliTom | **53.14%** | +0.48 | [holitom](../methods/holitom.md) |
| MDP3 | **53.06%** | +0.40 | [mdp3](../methods/mdp3.md) |
| AIM | **52.86%** | +0.20 | [aim](../methods/aim.md) |
| VideoITG | **52.86%** | +0.20 | [videoitg](../methods/videoitg.md) |
| *baseline* | *52.66%* | — | — |
| STTM | **51.72%** | −0.94 | [sttm](../methods/sttm.md) |
| VisionZip† | **40.09%** | −12.6 | [visionzip](../methods/visionzip.md) |
| FastV | **36.78%** | −15.9 | [fastv](../methods/fastv.md) |

† VisionZip's published backbone is LLaVA-1.5; the number above is that run
(other-backbones track). See its page.

## Reading these numbers

Four methods beat the backbone, all by **< 0.8 points** — within noise (~16
questions on 4,018). **On LLaVA-OV, efficiency methods are statistically
indistinguishable from the backbone.** FastV and VisionZip drop sharply because
they were run at the standardized 15% retention, which is far more aggressive than
their paper defaults (FastV's is r=0.5); their degradation is real (predictions
differ from baseline in 4,600+ samples), not a bug.

## Backbone-specific pitfalls (see [SETUP.md](../SETUP.md))

- Load with `attn_implementation="sdpa"`. **eager attention → 100% empty output.**
- DyCoke's builder defaults to flash-attn (not installed) → pass `sdpa`.
- Image-token span starts at index 14 (`qwen_2` preamble); the true visual-token
  count is `lengeh_vision_token`, set in `prepare_inputs_labels_for_multimodal`.

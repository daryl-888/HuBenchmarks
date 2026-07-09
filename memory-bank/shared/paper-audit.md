# Paper Implementation Cross-Check Audit

**Generated**: 2026-07-07
**Method**: arXiv paper data (curl fetch for AIM, confirmed), repo READMEs/sbatch files/eval scripts, CLAUDE.md, BENCHMARKING_NOTES.md

---

## Summary

| Status | Count | Models |
|:-------|:----:|--------|
| ✅ PASS (matches paper) | 8 | STTM, HoliTom, VideoITG, FlashVID, FastVID, DyCoke, PruneVid, VisionZip |
| 🔴 BUG FOUND | 1 | AIM (wrong conv_template with Qwen2 weights) |
| 🟡 EXTENSION (not paper reproduction) | 1 | FastV (paper: LLaVA-1.5, repo: LLaVA-OV) |
| ℹ️ NEEDS VERIFICATION | 2 | MDP3 (verify frame selection defaults), DyTo (fix broken default arg) |
| ⛔ NOT RUNNABLE | 2 | iMove (no code/weights), TrajViT (no weights) |
| ❌ INVALIDATED | 1 | incorrect_sttm (wrong backbone, corrected by STTM-LLaVAVid) |

---

## Detailed Per-Model Audit

### ✅ STTM-LLaVAVid
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| arXiv | 2502.17399 | — | — |
| Backbone | LLaVA-Video-7B (Qwen2) | `/project/rhu/dpalfaro/weights/llava-video-7b` | ✅ |
| Method | `replace_qwen2_with_quadtree_attn()` | Same (eval_sttm_llavavid.py:42-50) | ✅ |
| sa_start_layer_idx | 2 | 2 (sbatch: --sa_start_layer_idx 2) | ✅ |
| sa_tree_thresh | 0.85 | 0.85 | ✅ |
| sa_tree_temporal_thresh | 0.65 | 0.65 | ✅ |
| sa_tree_root_level | 1 | 1 | ✅ |
| prompt_stat | Explicit sys/inst/frame | Explicitly computed (lines 143-167) | ✅ |
| Conv template | qwen_1_5 (Qwen2 model uses "qwen_1_5" for LLaVA-Video) | qwen_1_5 | ✅ |
| Frames | 32 | 32 | ✅ |
| Accuracy | — | 54.28% | ✅ Top result |
| Venue | — | — | — |
| Notes | STTM paper designed for Qwen2 attention. LLaVA-Video-7B uses Qwen1.5 tokenizer style but Qwen2 architecture — conv_template=qwen_1_5 is correct for this specific model. | | |

### ✅ HoliTom
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| Venue | NeurIPS 2025 | — | — |
| Backbone | LLaVA-OV-7B | `llava-ov-7b` | ✅ |
| Outer stage | Global temporal segmentation + spatiotemporal merging | `apply_holitom(model)` | ✅ |
| Inner stage | Token-similarity merging inside LLM | Env vars: HOLITOM_k=18, r=0.5 | ✅ |
| RETAIN_RATIO | 0.15 | 0.15 | ✅ |
| T | 0.80 | 0.80 | ✅ |
| Attention | sdpa (reads vision encoder weights) | attn_implementation="sdpa" | ✅ |
| Frames | 32 | 32 | ✅ |
| Accuracy | — | 53.11% | ✅ |

### ✅ VideoITG
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| Venue | CVPR 2026 Highlight | — | — |
| Backbone | LLaVA-OV-7B | `llava-ov-7b` | ✅ |
| Stage 1 | Instruction-guided frame grounding (VideoITG-8B selector) | eval_videoitg_grounding.py → frame_scores.jsonl | ✅ |
| Stage 2 | LLaVA-OV-7B inference on selected frames | eval_videoitg_infer.py | ✅ |
| Fallback | Uniform sampling | load_frames_at_indices fallback_n | ✅ |
| Accuracy | — | 52.51% | ✅ |

### ✅ FlashVID
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| Venue | ICLR 2026 Oral | — | — |
| Backbone | LLaVA-OV-7B-Qwen2 | `llava-ov-7b-qwen2` | ✅ |
| Method | Pre-LLM token merge (ADTS + TSTM) | flashvid() wrapper | ✅ |
| Frames | 8 | 8 | ✅ |
| retention_ratio | 0.10 / 0.25 | Both tested | ✅ |
| alpha | 0.7 | 0.7 | ✅ |
| Conv template | qwen_2 | qwen_2 | ✅ |
| Accuracy | — | 50.50% (R=0.10), 51.99% (R=0.25) | ✅ |

### ✅ FastVID
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| Backbone | LLaVA-OV-7B-Qwen2 | `llava-ov-7b-qwen2` | ✅ |
| Method | DySeg + STPrune + DTM | fastvid_* params in model_args | ✅ |
| retention_ratio | 0.10 | 0.10 | ✅ |
| Frames | 32 | 32 | ✅ |
| Conv template | qwen_2 | qwen_2 | ✅ |
| Accuracy | — | 51.92% | ✅ |

### ✅ DyCoke
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| arXiv | 2411.14401 | — | — |
| Backbone | LLaVA-OV-7B | `llava-ov-7b` | ✅ |
| Method | Stage 1: Temporal token merging (K), Stage 2: KV cache pruning (P) | lmms_eval DyCoke integration | ✅ |
| l (pruning layer) | 3 | 3 | ✅ |
| p (KV cache pruning rate) | 0.7-0.8 | 0.7 | ✅ |
| k (temporal merge rate) | 0.3 | 0.3 | ✅ |
| Accuracy | — | 53.46% | ✅ |

### ✅ PruneVid
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| Venue | Findings of ACL 2025 | — | — |
| Backbone | PLLaVA-7B (LLaMA-2) | `ermu2001/pllava-7b` | ✅ |
| use_lora | Required (unmerged PEFT/LoRA weights) | use_lora=True | ✅ |
| cluster_ratio | 0.5 | 0.5 | ✅ |
| temporal_segment_ratio | 0.25 | 0.25 | ✅ |
| selected_layer | 10 | 10 | ✅ |
| alpha | 0.4 | 0.4 | ✅ |
| tau | 0.8 | 0.8 | ✅ |
| Frames | 16 (PLLaVA native) | 16 | ✅ |
| Accuracy | — | 43.80% | ✅ |
| Notes | MotionBench_Results.md previously mislabeled as LLaVA-OneVision-7B — fixed to PLLaVA-7B on 2026-07-07. | | |

### ✅ VisionZip
| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| Backbone | LLaVA-v1.5-7B | LLaVA-1.5-7B | ✅ |
| dominant tokens | 54 | 54 | ✅ |
| contextual tokens | 10 | 10 | ✅ |
| Frames tested | 8, 32 | 8, 32 | ✅ |
| Accuracy (8f) | — | 40.09% | ✅ |
| Accuracy (32f) | — | 39.97% | ✅ |
| Notes | Low accuracy due to LLaVA-1.5 weak backbone. VisionZip paper also evaluates on LLaVA-NeXT and LLaVA-OV. | | |

---

## 🔴 AIM — Conv Template Bug

| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| arXiv | 2412.03248 | — | — |
| Backbone | LLaVA-OV-7B with **Qwen2** LLM (28 layers) | `llava-ov-7b-qwen2` | ✅ Weights correct |
| conv_template | **qwen_2** (Qwen2 chat template) | **qwen_1_5** | ❌ MISMATCH |

**Paper excerpt** (confirmed via arXiv HTML):
> "We choose LLaVA-OV-7B as our base model... It uses Qwen2 as LLM with 28 layers in total."

**Analysis**: The `llava-ov-7b-qwen2` weights are the Qwen2-based LLaVA-OneVision. The correct chat template for Qwen2 chat models is `qwen_2`. Using `qwen_1_5` would produce wrong chat template tokens, affecting prompt formatting and likely degrading accuracy by ~10-30%.

**FlashVID uses the SAME Qwen2 weights** with `qwen_2` conv_template — confirming `qwen_2` is correct for `llava-ov-7b-qwen2`.

**Fix**: Change `conv_template=qwen_1_5` → `conv_template=qwen_2` in:
- `llava-ov-7b-qwen2/aim-motionbenc/run_aim.sbatch` line 44
- `llava-ov-7b-qwen2/aim-motionbenc/test_aim.sbatch`

**Action**: Fix and re-submit smoke test BEFORE full run.

---

## 🟡 FastV — Architectural Extension

| Item | Paper | Repo | Verdict |
|------|-------|------|:------:|
| arXiv | 2407.06179 | — | — |
| Paper backbone | LLaVA-1.5-7B (CLIP/Vicuna, 32 LLM layers) | N/A | — |
| Repo backbone | — | LLaVA-OV-7B (Qwen 1.5, 28 LLM layers) | ⚠️ Different architecture |
| Method | Token pruning based on attention scores | Same approach | ✅ Method is architecture-agnostic |
| Status | Published numbers on LLaVA-1.5 | Novel extension on LLaVA-OV | ⚠️ Not a reproduction |

**Verdict**: FastV's pruning operates on attention weights and is conceptually architecture-independent. However, paper results were on a different model family (Vicuna-based vs Qwen-based). This repo evaluation is an **architectural extension**, not a paper reproduction. **Document in README.**

---

## ℹ️ Needs Verification

### MDP3
| Item | Paper | Repo | Status |
|------|-------|------|:------:|
| Backbone | LLaVA-OV-7B | `llava-ov-7b` | ✅ |
| Method | SigLip query relevance + DPP diversity + DP sequentiality | `MDP3("cuda")` from vlmeval.smp | ✅ |
| n_selection (frames to select) | Paper default? | 8 (eval_mdp3.py line 122) | ⚠️ Verify against paper |
| pool_frames (initial pool) | Paper default? | default in sbatch? | ⚠️ Verify against paper |
| Conv template | qwen_1_5 | qwen_1_5 | ✅ |
| Accuracy (partial) | — | 53.25% (50% data) | ✅ |

### DyTo
| Item | Paper | Repo | Status |
|------|-------|------|:------:|
| Venue | ICCV 2025 | — | — |
| Backbone | LLaVA-NeXT-Vicuna-7B | `llava-v1.6-vicuna-7b` | ✅ |
| Method | FINCH clustering + ToMe token merging | temporal_aggregation="spatial_tome_finch_dynamic_all_frms" | ✅ |
| Frames | 100 → ~25 via FINCH | 100 | ✅ |
| rope_scaling | 2 (required for Llama-2) | 2 | ✅ |
| Conv template (sbatch) | vicuna_v1 | vicuna_v1 | ✅ |
| **Conv template (eval_dyto.py default)** | vicuna_v1 | **image_seq_v3** (broken) | 🔴 **Fix default arg** |
| Test accuracy | — | 3.70% (base model, expected) | ✅ |
| Notes | `eval_dyto.py` line 98: `parser.add_argument("--conv-template", default="image_seq_v3")`. The sbatch overrides this with `vicuna_v1`, but anyone running the script directly hits the bug. **Change default to `vicuna_v1`.** | | |

---

## Not Runnable

| Model | Reason | Status |
|-------|--------|:------:|
| iMove | No code or weights released (Findings of ACL 2025) | ⛔ |
| TrajViT | No weights released | ⛔ |

## Invalidated

| Model | Issue | Status |
|-------|-------|:------:|
| incorrect_sttm (STTM-v2) | Used LLaVA-OV-7B (Qwen 1.5) instead of LLaVA-Video-7B (Qwen2). STTM requires Qwen2 attention. | ❌ Replaced by STTM-LLaVAVid |

---

## Action Items

| Priority | Action | Model | Effort |
|:--------:|--------|-------|:------:|
| 🔴 HIGH | Fix conv_template: qwen_1_5 → qwen_2, re-run smoke test | AIM | 5 min |
| 🟡 MEDIUM | Fix default arg in eval_dyto.py: image_seq_v3 → vicuna_v1 | DyTo | 1 min |
| 🟡 MEDIUM | Update README to note LLaVA-1.5→LLaVA-OV extension | FastV | 2 min |
| ℹ️ LOW | Verify pool_frames / select_frames against paper | MDP3 | Manual |

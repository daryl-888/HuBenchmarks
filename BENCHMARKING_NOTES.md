# Benchmarking Notes — MotionBench on UH Carya

Honest accounting of what is correct, what is approximate, and what to watch for
when interpreting results.

---

## Setup

All models evaluate **LLaVA-OV-7B** on **MotionBench** (8,052 samples, ~4,034 NA,
4,018 scoreable) via the **lmms_eval** framework. Each subdirectory wraps its
model's patch around the same lmms_eval call, making the comparison
apples-to-apples at the framework level.

Cluster: UH Carya, single A40/Ada GPU, SLURM. No checkpointing — a job that dies
restarts from scratch.

---

## DyCoke (`dycoke-motionbenc/`)

### What is correct
- DyCoke's token compression (l=3, p=0.8, k=0.3) is applied before every
  forward pass via the standard DyCoke integration in lmms_eval.
- All 8,052 samples are attempted; results are scored with the NA-skip protocol.

### Caveats

**NFS stale handles (samples ~1168).**
Two videos on Carya have kernel-level stale NFS handles:
- `self-collected/bevgNkpc5dKYD8Un.mp4`
- `self-collected/l0w4V7yPdPJQQphx.mp4`

`os.path.exists` returns True but any read blocks the kernel in D-state
indefinitely — uncatchable by Python `try/except` or `SIGALRM`. The fix
(applied directly to Carya's `llava_onevision.py`) runs video loading in a
subprocess and kills it after 60 s, returning black frames and continuing.
Those samples score wrong but are not excluded. This is the correct behavior
for a benchmark — penalize the model, don't skip the sample.

**Tensor mismatch failures (samples 542, 1034).**
DyCoke's compression arithmetic fails on two samples with unusual token counts,
triggering lmms_eval's per-sample `except Exception` guard. Those samples
produce empty predictions and score 0. This is ~0.025% of the dataset and does
not meaningfully affect the final number.

---

## STTM (`sttm-motionbenc/`)

### What is correct
- STTM's `replace_qwen2_with_quadtree_attn()` runs before the model is loaded,
  replacing `Qwen2Model.forward` with STTM's spatiotemporal token-merging
  forward. The patch is active for every prefill.
- Hyperparameters match STTM paper defaults for LLaVA-OV-7B:
  `sa_start_layer_idx=2, sa_tree_thresh=0.85, sa_tree_temporal_thresh=0.65,
  sa_tree_root_level=1`.
- The evaluation runs through the same lmms_eval framework and MotionBench task
  as DyCoke, so the two are directly comparable.

### Compatibility fixes applied

STTM was written for `transformers==4.40.0.dev0` (an unreleased ByteDance fork).
The `sttm` conda env runs `4.45.2`. Five shims live in `run_sttm_eval.py`:

| Fix | Problem | Solution |
|-----|---------|----------|
| 1 | `LlavaQwenConfig` lacks `max_batch_size` | Wrap `load_pretrained_model` to inject `model.config.max_batch_size = 32` |
| 2 | `Qwen2Model._update_causal_mask` missing in 4.40 | Add a fallback implementation if the method is absent |
| 3 | `image_token_start_index` / `image_token_length` / `num_frame` never set | Hook `prepare_inputs_labels_for_multimodal` to infer and set them |
| 4 | `num_logits_to_keep` kwarg rejected by STTM's forward | Shim `Qwen2ForCausalLM.forward` to accept and discard it |
| 5 | `_sample` receives `prompt_stat=None` and returns `(sequences, runtime_dict)` | Initialize `prompt_stat={}` and strip `runtime_dict` before returning |

### Real uncertainty: Fix 3

This is the one fix whose correctness is inferred, not verified against STTM's
own output.

**What STTM's quadtree forward needs:**
- `self.image_token_start_index` — index in the embedding sequence where image
  tokens begin (i.e., the length of the system-prompt prefix in tokens).
- `self.image_token_length` — number of image tokens (must be exactly T×H×W).
- `self.num_frame` — T, the number of video frames.

**How STTM's original pipeline provides them:**
`generate()` receives an explicit `prompt_stat` dict with `sys`, `inst`, and
`frame` counts precomputed by the caller.

**How our hook provides them:**
We intercept `prepare_inputs_labels_for_multimodal` after it runs, and compute:
```
img_start     = position of first image token in input_ids
               (equals sys token count — should match prompt_stat["sys"])
img_tok_len   = new_embeds.shape[1] − (input_ids non-image tokens)
               (equals visual span — should match prompt_stat["video"])
n_frames      = inferred from images tensor shape
```

Then we truncate `img_tok_len` to the nearest multiple of `n_frames` to strip
the SigLIP CLS token (vision encoder emits T×196+1 tokens; the +1 is a CLS
that does not fit the T×H×W spatial grid STTM's quadtree expects).

**Why this is probably right but unverified:**
The algebraic equivalence holds — `img_mask.nonzero()[0][1]` gives the same
value as a sys-token counter. But this was never compared sample-by-sample
against STTM's own `prompt_stat` output. Edge cases (very short system prompts,
samples where the image token is not at position 0 of the visual span) could
silently misplace the quadtree window.

### What to watch for in results

| Observation | Likely cause |
|-------------|--------------|
| STTM accuracy ≈ baseline (no improvement or degradation) | Quadtree merging is working as expected — STTM trades a small accuracy drop for speed |
| STTM accuracy ≥ baseline by a large margin | Something is wrong; merging probably isn't triggering and the model ran as vanilla LLaVA-OV |
| STTM accuracy is drastically lower than baseline | `image_token_start_index` or `image_token_length` is wrong for many samples, causing the quadtree to merge the wrong tokens |
| Per-sample timing ~1.65 s/it | Consistent with quadtree merging active (prefill is faster than baseline) |
| Per-sample timing much slower | Merging overhead without benefit; check STTM params |

To confirm merging is active, look for `STTM patch applied:` in the `.out` file
and check that per-sample time is faster than a baseline run at the same
`max_new_tokens`.

---

## PruneVid (`prunevid-motionbenc/`)

### What is implemented

**`eval_prunevid.py`** (use this — not `eval_motionbench.py`):
- Loads LLaVA-OV-7B with the standard `load_pretrained_model`.
- **Stage 1** (SigLIP-level token pruning) is active: hooks SigLIP encoder
  layer 23's Q/K projections; after the full SigLIP forward pass, keeps the
  top `cluster_ratio` fraction of patch tokens by CLS-attention score; merges
  the discarded tokens into one weighted residual token.  Feature dim (1152)
  is unchanged; only token count is reduced (cluster_ratio=0.5 → ~50% kept).
- **Stage 2** (query-aware LLM pruning) is **not implemented**.  PruneVid's
  Stage 2 code (`models/pllava/modify_llama.py`) targets LLaMA attention;
  LLaVA-OV-7B uses Qwen2, which has a different attention signature.

### Why `eval_motionbench.py` is broken

The old script calls `tasks.eval.model_utils.load_llavaov_with_prunevid`, a
Carya-local custom function that replaces LLaVA-OV's SigLIP tower with a
wrong-dimensioned vision encoder.  The result is 384-dim patch features going
into an mm_projector that expects 1152-dim → matmul error on every sample.

PruneVid's public GitHub repo (`visual-ai/prunevid`) has no LLaVA-OV support
at all (only PLLaVA and LLaVA-NeXT-Video).

### Caveats

- Stage 2 is omitted; results will be slightly higher than full PruneVid
  because the LLM-side token pruning is not applied.
- Stage 1 algorithm is simplified vs. `pllava_prumerge.py`: the original
  iteratively updates each top-k cluster center by merging its k=32 nearest
  neighbors; our version keeps top-k as-is and adds one weighted residual.
  Token count reduction is identical; individual token values differ slightly.
- conda env is `prunevid`; `PYTHONPATH` still points to the PruneVid repo so
  the `tasks/` package is importable, but `eval_prunevid.py` does not import
  from it at runtime.

---

## Shared caveats (all models)

- **No checkpointing.** lmms_eval sorts samples by descending context length and
  processes them sequentially. A job that dies mid-run restarts from sample 1.
- **NA samples.** ~4,034 of 8,052 samples have ground-truth answer "NA" and are
  excluded from the accuracy denominator. Reported accuracy is over the ~4,018
  scoreable samples only.
- **Batch size = 1.** All video models run with `--batch_size 1` as required.
- **Baseline comparison.** A vanilla LLaVA-OV-7B run (no compression patch) is
  needed to interpret any model's number. Without it, we cannot separate
  MotionBench's inherent difficulty from the effect of token compression.

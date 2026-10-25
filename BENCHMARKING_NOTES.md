# Benchmarking Notes — MotionBench on UH Carya

Honest accounting of what is correct, what is approximate, and what to watch for
when interpreting results.

---

## Benchmark philosophy

Each model is evaluated using **its own native base model and inference pipeline**
— the exact setup the authors used, not forced onto a common base model.  This
means accuracy numbers across models are NOT directly apples-to-apples on the LLM
side, but they ARE apples-to-apples on:

- the dataset (MotionBench, same JSONL, same video files)
- the number of frames (32 for all)
- the scoring rule (NA-skip, letter-match regex)
- the hardware (single A40/Ada, SLURM)

To fairly compare two compression methods you need their respective baselines
(vanilla runs with the same base model, same num_frames).  Each directory should
eventually include a `run_baseline.sbatch` for that purpose.

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

## STTM v2 (`sttm-v2-motionbenc/`) — standalone replacement for sttm-motionbenc/

`eval_sttm.py` is a lmms_eval-free rewrite that fixes the two root causes that
made the lmms_eval-based run unreliable:

**Root cause 1 — `image_token_length` set wrong (einops crash at sample 1034):**
The lmms_eval wrapper inferred `image_token_length` from the embedding shape
after `prepare_inputs_labels_for_multimodal`, which added an extra
`image_newline` token (T×H×W + 1).  STTM's einops rearrange requires exactly
T×H×W.  Fix: build `prompt_stat` explicitly from token counts BEFORE the model
runs, and apply a pre-forward hook that truncates `image_token_length` to the
nearest multiple of `num_frame` at the point of use.

**Root cause 2 — STTM's bundled `llava/` shadows the installed package (384-dim
matmul crash on every sample):**
STTM's repo ships a partial `llava/` directory (`mm_utils.py`, `constants.py`,
`conversation.py`).  With `PYTHONPATH=/project/rhu/dpalfaro/code/STTM`, Python
finds STTM's `llava` first.  `process_images`, `tokenizer_image_token`, and
`conv_templates` all come from this partial copy, which processes images at a
different resolution and returns features of wrong dimension (384 instead of
1152), causing a matmul failure in the mm_projector.
Fix: in `apply_sttm_patch()`, after the STTM monkey-patch is imported, remove
STTM's root from `sys.path` and evict any cached `llava.*` entries from
`sys.modules`.  Subsequent `from llava.mm_utils import process_images` then
finds the env's fully-installed llava.

### Known unknowns

The pre-forward hook that corrects `image_token_length` has not been verified
sample-by-sample against STTM's own `prompt_stat` output.  Edge cases (unusual
aspect ratios, fractional tile counts) could silently produce wrong quadtree
windows.  If accuracy is anomalously low, add a print statement in the hook to
log `img_len → corrected` per sample and compare to expected T×H×W.

---

## PruneVid (`prunevid-motionbenc/`)

PruneVid is evaluated on **PLLaVA** — the base model it was designed and
published with — using PruneVid's own `load_pllava` / `pllava_answer` pipeline
unchanged.  The eval wrapper (`eval_prunevid.py`) only provides the MotionBench
data loop, subprocess video loading, and NA-skip scoring; it does not modify
PruneVid's model code.

### Conda env and PYTHONPATH

No `prunevid` env exists on Carya.  Use `dycoke11`.  PruneVid's own repo does
NOT have a top-level `llava/` directory so it does not shadow the installed
package (confirmed by `ls /project/rhu/dpalfaro/code/PruneVid/`).

### PLLaVA weights

PLLaVA-7B (`ermu2001/pllava-7b`) must be cached in
`/project/rhu/dpalfaro/cache/huggingface` before the job runs.  Download with
`TRANSFORMERS_OFFLINE=0` on a login node or via a short pre-fetch job.

### Caveats

- Results are for PLLaVA-7B + PruneVid, not LLaVA-OV-7B + PruneVid.  The
  baseline for this number is vanilla PLLaVA-7B on MotionBench (no pruning).
- PruneVid's pruning is fully active (Stage 1 vision-level and Stage 2
  LLM-attention-level) because we use PruneVid's unmodified `pllava_answer`.

---

## Shared caveats (all models)

- **No checkpointing.** A job that dies mid-run restarts from sample 1.
- **NA samples.** ~4,034 of 8,052 samples have ground-truth "NA" and are
  excluded from the accuracy denominator.  Accuracy = correct / scoreable.
- **Baseline required per model.** Each compression method needs a matching
  vanilla run (same base model, same num_frames, same env) to be interpretable.
- **num_frames = 32** for all models so frame-count is not a confound.
- **do_sample=False, max_new_tokens=16** for deterministic greedy decoding.

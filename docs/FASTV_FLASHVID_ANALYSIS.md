# FastV and FlashVID: why they underperform

**FastV collapses on LLaVA-OV — closed, not a bug.** **FlashVID underperforms
on Qwen3-VL — the reimplementation number is unverified; the authors'-code
re-run is in flight** with all three blocking bugs now fixed.

MotionBench, 4,018 scoreable questions, 32 frames, greedy decoding. LLaVA-OV-7B
baseline 52.66%, Qwen3-VL-8B baseline 62.52%. Resolution floor ±1.54 pt;
McNemar significance threshold χ² ≥ 3.84.

---

## Part 1 — FastV on LLaVA-OV: closed

FastV loses 15.9 points on LLaVA-OV, and the loss depends on none of the
obvious variables: not which tokens survive, not how many, not clip length.
This is a **density threshold**, not a ranking failure and not a bug.

### The retention range is flat

| Backbone | keep 10% | keep 15% | keep 25% | keep 50% | keep 75% | keep 100% |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA-OV-7B | 36.06 | 36.78 | 36.73 | 36.34 | 35.69 | 55.07\* |
| Qwen3-VL-8B | 56.92 | 59.01 | 60.60 | — | — | 62.52 |

\*986-question subset (baseline 54.36%), `keepall` only. Keep 50% is the
authors' own K=2/R=50% setting; keep 75% is K=2/R=25%; K=3/R=50% scores 35.94
(layer choice doesn't matter either).

LLaVA-OV spans 36.06–36.78 across the *entire* discarding range — a
0.72-point spread from keep 10% to keep 75% — and recovers only at keep 100%,
where nothing is discarded. Qwen3-VL, same code, responds to budget the whole
way (+3.68 from keep 10% to keep 25%), so this isn't an artifact of the
harness. At keep 15%, FastV answers **D** on ~47% of LLaVA-OV questions
against a 23.6% ground-truth rate — the signature of a model with the prompt
but no usable visual evidence. Paired against the backbone, 4,605 of 8,052
predictions differ, 1,113 broken against 475 fixed (χ² = 255.5).

### A four-arm study finds the cause

Identical backbone/prompt/frames/budget/decoding; only the token-selection
rule at the pruning layer varies (986-question subset):

| Arm | Rule | Overall | D-rate |
|---|---|---:|---:|
| *baseline* | *no reduction* | *54.36* | *20.7* |
| `keepall` | same path, discards nothing | **55.07** | 20.8 |
| `attention` | FastV as published | 39.25 | 49.4 |
| `random` | uniformly at random | 38.74 | 47.2 |
| `uniform` | evenly spread over 32 frames | 38.13 | 49.9 |

None of the three discarding arms differ significantly (χ² 0.16–1.54 pairwise).
Three conclusions: **not our code** (`keepall` returns the backbone exactly);
**not the ranking** (attention scores no better than random); **not frame
collapse** (`uniform` guarantees full coverage and doesn't help — every arm
touches all 32 frames on ~every sample). What remains is a threshold: below
it, discarding costs ~15 points regardless of which tokens survive; at 100%
the backbone is intact.

### Not a regime FastV was ever tested in

FastV's released repo (`/project/rhu/dpalfaro/code/FastV`) has exactly one
eval script, `eval_ocrvqa.sh` — static-image OCR-VQA on LLaVA-v1.5-7B.
Grepping their source for "video" or "frame" returns nothing. Their sweep:

```bash
model_path=llava-v1.5-7b
rank_list=(72 144 288 432)   # keep 12.5% / 25% / 50% / 75% of 576 tokens
Ks=(1 2 5 10 15 20)          # layer to prune at
--fast-v-image-token-length 576   # fixed: CLIP-ViT-L/14, one image
--fast-v-sys-length 36            # hardcoded LLaVA-1.5 prompt offset
```

Our keep-fractions match theirs; the *regime* doesn't. We ran LLaVA-OneVision
(SigLIP, not CLIP), 32 video frames, 6,273 visual tokens (11× their count),
motion questions instead of text-reading — four axes of difference at once.
Likely mechanism: FastV ranks tokens by layer-2 attention, which is a
well-posed target for one image but has to encode both *where* and *which
frame* across 32 near-duplicates, and motion signal is often a difference
across frames rather than a salient region in any one — a hard statistic to
extract that early. (This is the reasonable account, not something verified
layer-by-layer; the four-arm study above measured the consequence, not the
cause.) Their own most-published setting, K=2/R=50%, collapses identically
(36.34%) once transplanted into video — so it's the regime, not the setting.

### Two more checks, both negative

- **Duration.** A naive 3s split looks significant (asymmetry −5.20, χ²=16.28)
  but is doubly confounded: short clips are 33.7% Motion-related-Objects vs.
  11.7% long (category mix), and the backbone itself scores 62.5 short vs.
  49.4 long (floor effect). D-rate — immune to both confounds — moves the
  *opposite* direction (44.6% short vs. 47.8% long). Controlling for question
  type, the asymmetry drops to +0.09 (noise). Full derivation:
  `analysis/duration-split/`.
- **PruneVID-OV shares the signature.** `cluster_ratio=1.0` (keeps every
  token) scores 53.36%, +0.70 vs. backbone, not significant — the same
  recovery FastV's `keepall` shows. Two unrelated methods, same backbone,
  both intact at 100% and both collapsed the moment anything is discarded:
  this is a property of LLaVA-OV, not either port.

**Closed.** No threshold between 25% and 100% to locate — flat at ~36%
across the entire discarding range, including the authors' own operating
point, recovering only when nothing is discarded. Not duration-dependent.
Not a harness bug.

**Reproduce:**

```bash
cd analysis/fastv-selection-study
sbatch --export=ALL,SELECT=attention                          run_selection_study.sbatch
sbatch --export=ALL,SELECT=random                             run_selection_study.sbatch
sbatch --export=ALL,SELECT=uniform                             run_selection_study.sbatch
sbatch --export=ALL,SELECT=attention,FASTV_R=0.00,TAG=keepall run_selection_study.sbatch
python3 analysis/duration-split/split_by_duration.py --cut 3 --cut 5 --cut 10
python3 analysis/duration-split/duration_within_category.py --cut 5
```

Runs: `w2_fastv_run`, `s1_fastv_r{10,25,50,75}_run`, `s1_fastv_k3r50_run`,
`fastv_run1` (backbone), `w3_fastv_run`, `s3_fastv_r{10,25}_run`,
`qwen3vl_baseline_run1`, `fv_sel_{attention,random,uniform,keepall}`,
`s1_prunevid_c100_run`.

---

## Part 2 — FlashVID on Qwen3-VL: authors'-code re-run in flight

`w3_flashvid_run` — the 56.65% in the results tables (−5.87 vs. baseline) —
**never imports FlashVID's package**. It's a reimplementation missing
inner-LLM compression (`pruning_layer`, `llm_retention_ratio`), `expansion`,
segment scoring, and `attn_div` selection. Whether FlashVID is actually this
weak on Qwen3-VL, or that's our port, was unanswered — hence the re-run
against `flashvid()` directly, at the authors' published setting
(`retention_ratio=0.15, alpha=0.7, temporal_threshold=0.8,
token_selection_method=attn_div, min_segment_num=4, segment_threshold=0.9,
expansion=1.25, pruning_layer=28, llm_retention_ratio=0.1`). On **LLaVA-OV**,
by contrast, our port does call the real `flashvid()`, its omitted params
default to the authors' own published LLaVA-OV values, and the code fails
loudly (not silently) if misconfigured — accuracy-neutral there (53.31–53.36%)
is trustworthy.

### Three bugs found getting the authors' Qwen3-VL code to run

1. **Dtype crash — a regression in their release.** `modeling_qwen3_vl.py`'s
   vision forward does `hidden_states + pos_embeds` with no dtype cast;
   stock `transformers` casts `pos_embeds.to(hidden_states.dtype)` first.
   `fast_pos_embed_interpolate` returns float32 against a bf16 model → crash
   on 8,044/8,052 samples (0.00% accuracy, gate correctly failed it). Fixed
   with the one-line cast, confirmed by diffing against the stock code it was
   patched from.

2. **Hard FlashAttention-2 dependency, no working build on the cluster.**
   Their vision attention has no sdpa/eager fallback. A prebuilt wheel
   matching torch/CUDA/Python installed but failed on import (ABI mismatch).
   A "from source" `pip install` silently re-downloaded the same broken wheel
   — flash-attn's `setup.py` guesses a wheel URL and skips compiling unless
   `FLASH_ATTENTION_FORCE_BUILD=TRUE`. Forcing a real compile produced a
   genuinely different, working `flash_attn-2.8.3.post1`.

3. **`cache_position=None` crash — an environment mismatch, not a code bug.**
   Official FlashVID pins `transformers==4.57.3`; Carya's shared `qwen3vl` env
   runs `5.14.1`. In 5.14.1, `Qwen3VLModel.forward()` no longer takes
   `cache_position` as an explicit argument — it's folded into
   `**kwargs`, and can arrive `None`. FlashVID's LLM-pruning stage slices it
   unconditionally (`cache_position[keep_global_indexes]`) → crash, past both
   fixes above. **Fix:** a dedicated `flashvid_qwen3vl_official` env, cloned
   from `qwen3vl` and pinned to `transformers==4.57.3` per FlashVID's own
   `pyproject.toml`. (Cloning briefly dropped free disk on the shared
   `/project/rhu` volume from 12G to 7.6G — the same pressure that caused an
   earlier six-job outage — aborted and cleaned up once, then re-run
   successfully once the actual footprint was known to fit.) Two rounds of
   pip leaving stale, inconsistent `dist-info` metadata behind (`tokenizers`,
   `huggingface-hub` — real package files at one version, leftover metadata
   at another, because the clone's files were hardlinked from the original
   env's package cache) needed manual cleanup before the version pin actually
   took effect.

Debug run against the fixed env produced a real prediction (`'B'`) with no
crash — first clean execution of the authors' actual mechanism. Full
8,052-question run is submitted; once it gates (`check_run.py --vs-baseline
qwen3vl_baseline_run1`), this section closes with a real number to compare
against the reimplementation's −5.87.

**Reproducibility note carried from earlier re-runs:** a genuinely identical
run diverges by exactly 0 predictions; two runs that looked identical but
differed by 2.14% turned out to differ in model class and retention, not
prompt template as first suspected. Lesson: a run's stored parameter is not
proof of what it executed — behavioral divergence is the more reliable
witness. (Divergence scale, for calibration: 0 for nothing changed, 13–23 for
code-version drift, ~1,100+ for an actual retention change.)

**Reproduce:**

```bash
diff /project/rhu/dpalfaro/code/FlashVID/flashvid/modeling_qwen3_vl.py.bak \
     /project/rhu/dpalfaro/code/FlashVID/flashvid/modeling_qwen3_vl.py
sbatch analysis/upstream-faithful/build_flash_attn.sbatch
sbatch analysis/upstream-faithful/run_flashvid_qwen3vl_official_v2.sbatch
```

Runs: `w2_flashvid_run`, `s1_flashvid_r{10,25}_run`, `w3_flashvid_run`,
`s3_flashvid_official_run` (in flight, job 7987340).

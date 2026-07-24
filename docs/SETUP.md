# Setup

Everything needed to run the benchmark on your own machine or cluster.

## 1. Paths

Every script uses hardcoded paths from our cluster (UH Carya). Redirect them by
editing **[`config/paths.sh`](../config/paths.sh)** (copy to `paths.local.sh`
first) and `source`-ing it. It defines `$W_LLAVA_OV`, `$MOTIONBENCH_META`,
`$HUVLLM_RESULTS`, etc.

## 2. Dataset — MotionBench

Layout expected under `$MOTIONBENCH`:

```
MotionBench/
├── video_info.meta.jsonl      # 8,052 records; each has qa[0].question / answer / question_type
├── self-collected/            # .mp4 files
└── public-dataset/            # .mp4 files
```

8,052 total samples · 4,018 scoreable · 4,034 `NA` (unanswerable, skipped in scoring).

## 3. Model weights

Download these HuggingFace snapshots into `$HUVLLM_WEIGHTS/<dir>`:

| Config var | HF repo | Backbone role |
|---|---|---|
| `$W_LLAVA_OV` | `lmms-lab/llava-onevision-qwen2-7b-ov` | Stage 1 backbone (Qwen2 LLM) |
| `$W_QWEN3VL` | `Qwen/Qwen3-VL-8B-Instruct` | Stage 3 backbone (Qwen3 LLM) |
| `$W_PLLAVA` | `ermu2001/pllava-7b` | PruneVID (PEFT/LoRA — load with `use_lora=True`) |
| `$W_VICUNA` | `liuhaotian/llava-v1.6-vicuna-7b` | DyTo |
| `$W_LLAVA15` | `liuhaotian/llava-v1.5-7b` | VisionZip |
| `$W_LLAVA_VIDEO` | `lmms-lab/LLaVA-Video-7B-Qwen2` | STTM-LLaVAVid (other-backbones) |
| `$W_VIDEOITG` | `nvidia/VideoITG-8B` | VideoITG grounding stage |

## 4. Method source repositories

Each method wraps its authors' original code, cloned into `$HUVLLM_CODE/` and
patched. The diffs we applied are in [`patches/`](../patches/). Clone:

| Var | Repo |
|---|---|
| `$SRC_DYCOKE` | github.com/KD-TAO/DyCoke |
| `$SRC_FASTV` | github.com/pkunlp-icler/FastV |
| `$SRC_HOLITOM` | github.com/cokeshao/HoliTom |
| `$SRC_MDP3` | (ICCV 2025 MDP3 frame selector) |
| `$SRC_VISIONZIP` | github.com/dvlab-research/VisionZip |
| `$SRC_PRUNEVID` | github.com/Visual-AI/PruneVid |
| `$SRC_DYTO` | github.com/Yunkang-Sun/DyTo (vendors its own `llava` as `dyto.llava`) |

## 5. Conda environments

> **Exact pins are in [`config/envs/`](../config/envs/)** — `pip freeze` captured
> from the environments that produced the published results. Install with
> `pip install -r config/envs/<env>.txt` rather than reconstructing from the prose
> below. The versions listed here were **verified against the live environments**
> on 2026-07-24 (earlier drafts of this table were wrong).

Methods need **different** environments — they pin conflicting `transformers`
versions. Key constraints (full detail in each method page):

| Env | transformers | Used by |
|---|---|---|
| `dycoke11` | **4.40.0** (torch 2.12.0+cu130) | DyCoke, FastV, AIM, FlashVID, STTM, VisionZip, VideoITG |
| `holitom` | **4.45.2** (exact, torch 2.12.0+cu130) | HoliTom |
| `mdp3` | 4.40 + DPP/vlmeval deps | MDP3 |
| `prunevid` | — | PruneVID (PLLaVA) |
| `dyto` | **4.40.0** (torch 2.12.0+cu130) | DyTo |
| `qwen3vl` | **5.14.1**, torch 2.6.0+cu124 | **all Qwen3-VL** (only env with `Qwen3VLForConditionalGeneration`) |

**Do not** `pip install -r requirements.txt` in `dycoke11` / the sttm env — those
were cloned from a ByteDance internal dump with private packages.

### The `PYTHONNOUSERSITE` trap
On a shared cluster, `pip install` may land packages in `~/.local`, which every
sbatch ignores (`PYTHONNOUSERSITE=1`) — so a job fails on a package you "installed".
Always install into the env explicitly and verify:
```bash
SP=$HUVLLM_ENVS/<env>/lib/python3.10/site-packages
pip install --no-cache-dir --target=$SP <pkg>
PYTHONNOUSERSITE=1 $HUVLLM_ENVS/<env>/bin/python3 -c "import <pkg>; print(<pkg>.__file__)"
# printed path must be under the ENV, not ~/.local
```

## 6. Two backbone-specific gotchas

- **eager attention breaks LLaVA-OV generation** (100% empty output). Load with
  `attn_implementation="sdpa"`. DyCoke's builder defaults to flash-attn, which
  isn't installed — pass `sdpa` explicitly.
- **Qwen3-VL re-samples frames.** `Qwen3VLVideoProcessor` has
  `do_sample_frames=True, fps=2`, so it ignores your `--num_frames` unless you
  pass `do_sample_frames=False` to the processor.

## 7. Hardware used for the published results

All numbers in [RESULTS.md](RESULTS.md) were produced on the **UH Carya** cluster:

| | |
|---|---|
| GPU | **NVIDIA Ada-generation, ~44.4 GiB usable per device** (`gpu:ada`, 2 per node) |
| Nodes used | `compute-9-[3,4,6]`, `compute-10-[8,9]` |
| Per job | 1 GPU · 8 CPU cores · 124 GB RAM |
| Batch size | **1** (required for video models) |
| Precision | bfloat16 (Qwen3-VL), float16 (LLaVA-family) |

**Runtimes** (full 8,052-sample run): LLaVA-OV methods 1.8–7.4 h; Qwen3-VL ~8–10 h
(it carries ~11,664 visual tokens per sample vs ~6,273 for LLaVA-OV).

**Memory notes for replicators**
* ~44 GiB was enough for every LLaVA-OV method at 32 frames.
* Qwen3-VL-8B at 32 frames fits, but with less headroom.
* **DyTo at its paper default of 100 frames does not fit** — it OOM'd needing an
  extra ~5 GiB. We run it at 32 frames (its FINCH stage selects ~25 regardless).
  On an 80 GB card the paper default should be reachable.
* Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

**Reproducibility caveat.** Re-running the same configuration reproduced
**bit-identical** predictions across different nodes of this same GPU generation
(verified: 0/8052 differences on three separate pairs). We have **not** tested
reproduction on a different GPU generation. Because floating-point reduction order
can differ across architectures and `argmax` can flip on near-ties, replicators on
other hardware should expect *close* but not necessarily identical numbers. See
[DETERMINISM_AND_VALIDITY.md](DETERMINISM_AND_VALIDITY.md).

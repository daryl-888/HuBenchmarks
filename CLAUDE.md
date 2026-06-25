# HuBenchMarks

Evaluating video LLMs on MotionBench on UH Carya. Each subdirectory is one model.

When working on this project, follow these rules and use the reference below.

---

## Rules

- Never run inference on the login node — all jobs go through SLURM (`sbatch`)
- Never `pip install -r requirements.txt` in the sttm env — it is a ByteDance internal dump with private packages
- Don't put bare functions at module level in any file lmms_eval imports — the model registry calls `issubclass()` on every module member without checking if it's a class first, which crashes the job
- Batch size must be 1 for video models
- If DyCoke code is re-cloned or reset, re-apply the `load_video` patch (see below)
- All `utils.py` files must return `[]` (not a fallback path) when a video is missing — never `return [os.path.join(...)]` unconditionally
- When uploading directories to Carya via `scp`, create the destination first: `ssh dpalfaro@carya.rcdc.uh.edu "mkdir -p /path"`, then `scp -r`
- lmms_eval has no checkpointing — if a job dies mid-run, it restarts from scratch

---

## Cluster: UH Carya

SSH: `ssh -l dpalfaro carya.rcdc.uh.edu`

### Paths on Carya

| Path | What |
|------|------|
| `/project/rhu/dpalfaro/code/DyCoke` | DyCoke source + patched lmms_eval |
| `/project/rhu/dpalfaro/code/dycoke-motionbenc` | synced from this repo |
| `/project/rhu/dpalfaro/code/STTM` | STTM source (github.com/HYUNJS/STTM) |
| `/project/rhu/dpalfaro/code/sttm-motionbenc` | synced from this repo |
| `/project/rhu/dpalfaro/code/PruneVid` | PruneVid source (patched) |
| `/project/rhu/dpalfaro/code/prunevid-motionbenc` | synced from this repo |
| `/project/rhu/dpalfaro/code/HoliTom` | HoliTom source (patched) |
| `/project/rhu/dpalfaro/code/holitom-motionbenc` | synced from this repo |
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | LLaVA-OV-7B model weights |
| `/project/rhu/dpalfaro/weights/pllava-7b` | PLLaVA-7B weights (PruneVid) |
| `/project/rhu/dpalfaro/conda/envs/dycoke11` | main conda env |
| `/project/rhu/dpalfaro/conda/envs/sttm` | STTM env (cloned from dycoke11) |
| `/project/rhu/dpalfaro/conda/envs/prunevid` | PruneVid env |
| `/project/rhu/dpalfaro/conda/envs/holitom` | HoliTom env (transformers==4.45.2) |
| `/project/rhu/dpalfaro/cache/huggingface` | HF cache |
| `/project/rhu/dpalfaro/results` | job output + eval results |
| `/project/rhu/dpalfaro/sttm_features` | STTM pre-extracted features |
| `/project/rhu/dpalfaro/bad_videos` | NFS-broken videos, skipped at runtime |
| `/project/rhu/dpalfaro/sample_videos` | staged videos for local SCP |
| `/project/rhu/MotionBench_Data/MotionBench` | videos + metadata JSONL |

### Uploading to Carya

```bash
scp <file> dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/<subdir>/
# directory: create it first
ssh dpalfaro@carya.rcdc.uh.edu "mkdir -p /project/rhu/dpalfaro/code/<subdir>"
scp -r <local_dir>/ dpalfaro@carya.rcdc.uh.edu:/project/rhu/dpalfaro/code/<subdir>
```

---

## Critical Patch on Carya (NFS hang fix)

**Applied directly to `/project/rhu/dpalfaro/code/DyCoke/lmms_eval/models/llava_onevision.py` — not in any upstream repo.**

Some MotionBench videos have stale NFS handles. `os.path.exists` returns True but any read blocks the kernel indefinitely in D-state — can't be caught by `try/except` or `SIGALRM`. Only fix is subprocess isolation with hard kill on timeout.

Patched `load_video`:

```python
def load_video(self, video_path, max_frames_num):
    import multiprocessing as _mp
    import queue as _queue
    import numpy as _np
    import logging as _logging

    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0))
            total = len(vr)
            indices = np.linspace(0, total - 1, n, dtype=int).tolist()
            frames = vr.get_batch(indices).asnumpy()
            q.put(('ok', frames))
        except Exception as e:
            q.put(('error', str(e)))

    path = video_path if type(video_path) == str else video_path[0]
    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(path, max_frames_num, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)   # get WHILE child is alive, not after join
    except _queue.Empty:
        proc.kill()
        proc.join(timeout=5)   # D-state child ignores SIGKILL; don't wait forever
        _logging.warning(f'load_video: timeout (NFS stale?), skipping with black frames: {path}')
        return _np.zeros((max_frames_num, 336, 336, 3), dtype=_np.uint8)
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == 'error':
        _logging.warning(f'load_video: decode error, skipping with black frames: {path} — {data}')
        return _np.zeros((max_frames_num, 336, 336, 3), dtype=_np.uint8)
    return data
```

**Critical:** `raise RuntimeError` was replaced with a black-frames fallback. The old version that raised on timeout/error would crash the entire lmms_eval job (no per-sample try/except), causing the reproducible freeze at sample 1168. Now a bad video scores wrong but the job continues.

`q.get` must come before `proc.join` — for a 22MB numpy array the Queue feeder thread hasn't finished by the time join returns, so `get_nowait` after join will miss the data.

Re-apply this patch if DyCoke is re-cloned or reset.

---

## MotionBench

- Metadata: `/project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl`
- 8,052 samples; ~4,034 are "NA" (unanswerable); 4,018 scoreable
- 5,385 unique videos across `self-collected/` and `public-dataset/`

### Bad videos (NFS stale handles)

Both trigger at ~sample 1168 in lmms_eval's sorted order. The `load_video` patch returns black frames and continues — jobs no longer freeze here.

| File | Notes |
|------|-------|
| `self-collected/bevgNkpc5dKYD8Un.mp4` | moved to `/project/rhu/dpalfaro/bad_videos/`; 3 JSONL entries |
| `self-collected/l0w4V7yPdPJQQphx.mp4` | confirmed stale 2026-05-28 (DyCoke job 6990458) |

Scan completed 2026-05-27: 5,385 scanned, found 1 (missed l0w4V7yPdPJQQphx.mp4). Re-scan: `sbatch dycoke-motionbenc/scan_videos.sbatch`.

---

## Repo Structure

```
dycoke-motionbenc/      DyCoke  (l=3, p=0.8, k=0.3)
sttm-motionbenc/        STTM (two-step: extract features → inference)
holitom-motionbenc/     HoliToM
imove-motionbenc/       iMove
prunevid-motionbenc/    PruneVid
trajvit-motionbenc/     TrajViT
videoitg-motionbenc/    VideoITG
```

Each dir:
- `run_<model>.sbatch` — full eval
- `test_<model>.sbatch` — 10-sample smoke test
- `run_baseline.sbatch` — vanilla LLaVA-OV-7B
- `tasks/motionbench/` — lmms_eval task (`motionbench.yaml` + `utils.py`)

### STTM evaluation approach

STTM patches Qwen2's LLM attention with quadtree merging via `replace_qwen2_with_quadtree_attn()`. The vision encoder runs normally — efficiency comes from LLM layers only.

`run_sttm_eval.py` applies the patch before lmms_eval loads the model, then hands off to lmms_eval CLI (same as DyCoke, no pre-extraction). PYTHONPATH must put STTM first (for the patch module), then DyCoke (for lmms_eval + NFS-fixed llava_onevision). Uses `dycoke11` env.

`run_sttm.sbatch` / `test_sttm.sbatch` are single-step jobs.

STTM's original repo uses a custom two-step pipeline (`video_feat_llavavideo.py` → `eval_vidqa_by_feat_llavavideo.py`) with pre-extracted features for its own datasets. The lmms_eval wrapper here is simpler and produces identical accuracy.

Do not run `pip install -r requirements.txt` in any env — ByteDance internal dump with private packages.

---

## utils.py Pattern

All `utils.py` files are identical in structure — no MOTION_CATEGORIES, returns `[]` (not a broken path) if video is missing:

```python
import re
import os
from loguru import logger as eval_logger

VIDEO_BASE_PATH = "/project/rhu/MotionBench_Data/MotionBench"

def motionbench_doc_to_visual(doc):
    video_path = doc["video_path"]
    for subdir in ("self-collected", "public-dataset"):
        full_path = os.path.join(VIDEO_BASE_PATH, subdir, video_path)
        if os.path.exists(full_path):
            return [full_path]
    return []

def motionbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["qa"][0]["question"]
    post_prompt = "\nAnswer with the option's letter from the given choices directly."
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)
    return f"{question}{post_prompt}"

def motionbench_doc_to_target(doc):
    return doc["qa"][0]["answer"]

def motionbench_process_results(doc, results):
    prediction = results[0] if results else ""
    match = re.search(r'\b([A-D])\b', prediction.upper())
    pred_letter = match.group(1) if match else prediction.strip().upper()[:1]
    ground_truth = doc["qa"][0]["answer"].strip().upper()
    if ground_truth == "NA":
        return {"motionbench_accuracy": None, "category": doc.get("question_type", "Unknown")}
    return {"motionbench_accuracy": int(pred_letter == ground_truth), "category": doc.get("question_type", "Unknown")}

def motionbench_aggregate_results(results):
    valid = [r for r in results if r is not None]
    if not valid:
        return 0.0
    correct = sum(valid)
    total = len(valid)
    skipped = len(results) - total
    eval_logger.info(f"Accuracy: {correct}/{total} = {correct / total:.4f} ({skipped} NA samples excluded)")
    return correct / total
```

---

## lmms_eval Notes

- Sorts by `-len(tok_encode(context))` — longest context first, no checkpointing
- If the job dies, restart from scratch
- Don't put bare functions at module level in any file lmms_eval imports — the model registry calls `issubclass()` on every module member without checking if it's a class first
- Batch size must be 1 for video models

---

## Conda Envs

| Env | Path | Used for |
|-----|------|----------|
| `dycoke11` | `/project/rhu/dpalfaro/conda/envs/dycoke11` | DyCoke, STTM baseline |
| `sttm` | `/project/rhu/dpalfaro/conda/envs/sttm` | STTM (cloned from dycoke11) |
| `prunevid` | `/project/rhu/dpalfaro/conda/envs/prunevid` | PruneVid / PLLaVA-7B |
| `holitom` | `/project/rhu/dpalfaro/conda/envs/holitom` | HoliTom |
| `videoitg` | `/project/rhu/dpalfaro/conda/envs/videoitg` | VideoITG |
| `flashvid` | `/project/rhu/dpalfaro/conda/envs/flashvid` | FlashVID (clone dycoke11, add flash-attn + flashvid) |

Create sttm (if missing): `conda create --name sttm --clone dycoke11`

---

## HoliTom Setup Notes

HoliTom uses LLaVA-OV-7B with a custom hierarchical token pruning patch.
Source: `/project/rhu/dpalfaro/code/HoliTom`
PYTHONPATH: `HoliTom/LLaVA-NeXT:HoliTom`

### transformers version

The holitom conda env must have transformers==4.45.2 (not ≥4.47).
HoliTom's `holitom/modeling_qwen2.py` patches Qwen2 attention and was written for 4.45.
Newer versions added symbols it doesn't expect (`FlashAttentionKwargs`, etc.).

To install:
```bash
/project/rhu/dpalfaro/conda/envs/holitom/bin/pip install "transformers==4.45.2"
```

Then hand-patch the rope_parameters None guard in the installed qwen2 modeling file
(see `patches/README.md` for exact lines).

### Source file patches

Four files in the HoliTom repo need patching. See `patches/README.md` and run
`patches/collect_patches.sh` on Carya to get the current fixed versions.

Key changes:
- `holitom/modeling_qwen2.py` — stubs for all transformers 4.47+ symbols not in 4.45.2
- `LLaVA-NeXT/llava/model/multimodal_encoder/siglip_encoder.py` — removed `device_map=device_map` from inner `SigLipVisionModel.from_pretrained` to avoid nested meta device context
- `LLaVA-NeXT/llava/__init__.py` — LlavaLlamaForCausalLM import wrapped in try/except
- `LLaVA-NeXT/llava/model/builder.py` — `low_cpu_mem_usage=True` kept (required by device_map)

`eval_holitom.py` calls `load_pretrained_model` WITHOUT `device_map="auto"`, then
calls `model = model.cuda()` explicitly.

---

## PruneVid Setup Notes

PruneVid uses PLLaVA-7B (not LLaVA-OV-7B) as its backbone + VTP (Video Token Pruning).
Source: `/project/rhu/dpalfaro/code/PruneVid`
PYTHONPATH: `PruneVid`

### Checkpoint format

`/project/rhu/dpalfaro/weights/pllava-7b` contains unmerged PEFT/LoRA format.
`load_pllava` MUST be called with `use_lora=True, weight_dir=args.model_path`.
Using `use_lora=False` loads only 291 base weights and silently drops 128 LoRA deltas
→ garbage output at 43% accuracy (barely above random).

### Source file patches

Two files need patching. See `patches/README.md`.
- `models/pllava/llama.py` — custom dataclass + `getattr(config, 'mlp_bias', False)` fix
- `models/pllava/modeling_pllava.py` — shape mismatch fix in VTP token selection

---

## Backbone / LLM version map

| Backbone weights | LLM | Conv template | Used by |
|---|---|---|---|
| `llava-ov-7b` | Qwen 1.5 | `qwen_1_5` | DyCoke, HoliTom, VideoITG, OV baseline |
| `llava-video-7b` | Qwen 2 | `qwen_2` | STTM (primary run), Video baseline |
| `llava-ov-7b-qwen2` | Qwen 2 | `qwen_2` | FlashVID |
| `pllava-7b` | LLaMA-2 | — | PruneVid, PLLaVA baseline |

Note: `qwen_1_5` is the conversation template used with Qwen1.5-based LLaVA-OV weights.
FlashVID uses a Qwen2-based LLaVA-OV — different weights from the existing `llava-ov-7b`.
Weights: `lmms-lab/llava-onevision-qwen2-7b-ov` → `/project/rhu/dpalfaro/weights/llava-ov-7b-qwen2`

---

## Results Summary (2026-06)

| Model | Backbone | LLM | Frames | Accuracy | Notes |
|-------|----------|-----|--------|----------|-------|
| LLaVA-OV-7B baseline | LLaVA-OV | Qwen 1.5 | 32 | 52.69% | DyCoke sbatch `--no_pruning` |
| LLaVA-Video-7B baseline | LLaVA-Video | Qwen 2 | 32 | 56.39% | Strongest baseline |
| DyCoke | LLaVA-OV | Qwen 1.5 | 32 | 53.46% | l=3, p=0.7 (p=0.8 identical) |
| STTM | LLaVA-Video | Qwen 2 | 32 | 54.28% | Quadtree LLM attn |
| HoliTom | LLaVA-OV | Qwen 1.5 | 32 | 53.11% | RETAIN_RATIO=0.15 |
| VideoITG | LLaVA-OV | Qwen 1.5 | 32 | 52.51% | Two-stage grounding |
| PruneVid | PLLaVA-7B | LLaMA-2 | 32 | 44.00% | VTP; 16f=43.80%, backbone bottleneck |
| PLLaVA-7B baseline | PLLaVA-7B | LLaMA-2 | 16 | 43.35% | Confirms backbone |
| FlashVID (retention=0.10) | LLaVA-OV Qwen2 | Qwen 2 | 8 | 50.50% | ICLR 2026 Oral; pre-LLM merge; alpha=0.7; below baseline |
| FlashVID (retention=0.25) | LLaVA-OV Qwen2 | Qwen 2 | 8 | 51.99% | Same config, higher retention; still below baseline |
| VisionZip | LLaVA-1.5-7B | CLIP | 8 | 40.09% | dominant=54, contextual=10; LLaVA-1.5 backbone bottleneck |

VideoITG per-category highlights: Motion-related Objects 70.1%, Repetition Count 26.5%.
TrajViT and iMove: not runnable (retrieval-only / no weights released).

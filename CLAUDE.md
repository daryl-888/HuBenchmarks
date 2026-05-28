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
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | model weights |
| `/project/rhu/dpalfaro/conda/envs/dycoke11` | main conda env |
| `/project/rhu/dpalfaro/conda/envs/sttm` | STTM env (cloned from dycoke11) |
| `/project/rhu/dpalfaro/cache/huggingface` | HF cache |
| `/project/rhu/dpalfaro/results` | job output + eval results |
| `/project/rhu/dpalfaro/sttm_features` | STTM pre-extracted features |
| `/project/rhu/dpalfaro/bad_videos` | NFS-broken videos, skipped at runtime |
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

### Bad video

`self-collected/bevgNkpc5dKYD8Un.mp4` — NFS stale handle, moved to `/project/rhu/dpalfaro/bad_videos/`. Affects 3 JSONL entries. Not on HF Hub. Accuracy impact ≤ 0.05%.

Scan completed 2026-05-27: 5,385 scanned, 1 bad. Re-scan: `sbatch dycoke-motionbenc/scan_videos.sbatch`.

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

| Env | Path |
|-----|------|
| `dycoke11` | `/project/rhu/dpalfaro/conda/envs/dycoke11` |
| `sttm` | `/project/rhu/dpalfaro/conda/envs/sttm` |

Create sttm (if missing): `conda create --name sttm --clone dycoke11`

# Standalone eval template — for every new model

The right structure for each `<model>-motionbenc/` directory.
No lmms_eval wrapper. No monkey-patching around framework mismatches.
The model runs exactly as its authors intended.

---

## Directory layout

```
<model>-motionbenc/
    eval_<model>.py        — standalone eval script (the only file that matters)
    run_<model>.sbatch     — full 8052-sample run
    test_<model>.sbatch    — 10-sample smoke test
```

No `tasks/` directory. No `utils.py`. No lmms_eval task yaml.

---

## What eval_<model>.py must do

```python
# 1. Apply the model's patch BEFORE loading weights
apply_patch(args)

# 2. Load model via the model's OWN loader (not lmms_eval's wrapper)
model, tokenizer, processor = load_model(args.model_path)

# 3. For each MotionBench sample:
for sample in samples:
    # 3a. Load video — ALWAYS subprocess-isolated (NFS safety)
    frames = load_frames(video_path, args.num_frames)

    # 3b. Build inputs using the model's OWN preprocessing
    #     Count token spans explicitly (sys / image / inst) if the model needs them
    inputs = build_inputs(frames, question, tokenizer, processor, model)

    # 3c. Run inference using the model's OWN generate() — no wrappers
    output = model.generate(**inputs, do_sample=False, max_new_tokens=16)

    # 3d. Decode only the newly generated tokens
    generated = output[:, inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    # 3e. Score
    score = score_prediction(prediction, sample["answer"])
```

---

## Scoring (identical for all models)

```python
def score_prediction(prediction: str, ground_truth: str):
    gt = ground_truth.strip().upper()
    if gt == "NA":
        return None                          # excluded from denominator
    match = re.search(r"\b([A-D])\b", prediction.upper())
    pred = match.group(1) if match else prediction.strip().upper()[:1]
    return int(pred == gt)
```

NA samples are skipped. Accuracy = correct / scoreable (not / total).

---

## Video loading (copy this verbatim)

```python
def load_frames(video_path: str, num_frames: int):
    """Subprocess-isolated. Returns list of PIL Images or raises RuntimeError."""
    import multiprocessing as _mp, queue as _queue

    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0))
            indices = np.linspace(0, len(vr) - 1, n, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            q.put(("ok", frames))
        except Exception as e:
            q.put(("error", str(e)))

    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frames, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill(); proc.join(timeout=5)
        raise RuntimeError(f"timeout (NFS stale?): {video_path}")
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill(); proc.join(timeout=5)
    if status == "error":
        raise RuntimeError(f"decode error: {video_path} — {data}")

    from PIL import Image
    return [Image.fromarray(f) for f in data]
```

Missing / stale videos: `load_frames` raises `RuntimeError`. The caller's
`try/except` logs a warning and records an empty prediction (scored wrong).
**Never skip the sample silently — always record it so the denominator is right.**

---

## Video path lookup (identical for all models)

```python
VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"

def find_video(video_path: str):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None   # caller records empty prediction, not skip
```

---

## sbatch skeleton

```bash
#!/bin/bash
#SBATCH -J <model>_motionbench
#SBATCH -o /project/rhu/dpalfaro/results/<model>_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/<model>_%j.err
#SBATCH -N 1 -n 8 --gpus=ada:1 --mem=124G -t 24:00:00
#SBATCH --mail-type=END --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

export PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export PYTHONPATH=/project/rhu/dpalfaro/code/<MODEL_REPO>:$PYTHONPATH
export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
export TRANSFORMERS_OFFLINE=1

cd /project/rhu/dpalfaro

/project/rhu/dpalfaro/conda/envs/<env>/bin/python3 \
    /project/rhu/dpalfaro/code/<model>-motionbenc/eval_<model>.py \
    --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
    --meta_path  /project/rhu/MotionBench_Data/MotionBench/video_info.meta.jsonl \
    --output_dir /project/rhu/dpalfaro/results/<model>_run1 \
    --num_frames 32
```

Test version: add `--limit 10` and shorten `-t` to `1:00:00`.

---

## Rules

- **PYTHONPATH**: include only what the model actually needs. Adding DyCoke
  for lmms_eval pulls in a patched llava_onevision that is specific to DyCoke
  and may interfere with other models' preprocessing.
- **No compatibility shims** unless you understand exactly what they change and
  have verified the result is the same as the model's original output.
- **Never modify** a model's source repo on Carya to make benchmarking work.
  Patches to source (like the DyCoke NFS fix) are the exception, not the rule,
  and must be documented.
- **Same num_frames** (32) for all models so results are comparable.
- **do_sample=False, max_new_tokens=16** for all models — deterministic greedy,
  short enough that the model must commit to a letter.
- **Upload the results.jsonl** (per-sample predictions) alongside summary.json
  so you can recompute accuracy by category without re-running.

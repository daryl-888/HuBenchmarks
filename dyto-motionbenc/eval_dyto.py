#!/usr/bin/env python3
"""
DyTo x MotionBench — Dynamic Token Merging for Zero-Shot Video Understanding (ICCV 2025).

DyTo applies two complementary strategies at inference time (no training):
  1. FINCH-based hierarchical frame clustering → selects ~25 representative frames from 100
  2. ToMe (Token Merging) with dynamic per-frame merge ratio → constrains total tokens to ~3,680

The temporal aggregation is triggered by a single keyword arg to model.generate():
    temporal_aggregation="spatial_tome_finch_dynamic_all_frms"

Source:     https://github.com/Jam1ezhang/DYTO
            /project/rhu/dpalfaro/code/DYTO

Backbone: LLaVA-NeXT Vicuna-7B (Llama-2 LLM, CLIP vision encoder).
  NOT LLaVA-OV — different weights from the rest of the benchmark.
  Weights: /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b
  Conv template: image_seq_v3 (DyTo-specific)
  RoPE scaling factor: 2 (required for Llama-2 context extension)

--- ONE-TIME SETUP ON CARYA (login node) ---

1. Clone DyTo:
   cd /project/rhu/dpalfaro/code
   git clone https://github.com/Jam1ezhang/DYTO

2. Download LLaVA-NeXT Vicuna-7B weights:
   huggingface-cli download liuhaotian/llava-v1.6-vicuna-7b \\
       --local-dir /project/rhu/dpalfaro/weights/llava-v1.6-vicuna-7b

3. Create conda env (fresh — DyTo needs torch==2.2.0, transformers==4.38.2):
   conda create -n dyto python=3.10 -y
   /project/rhu/dpalfaro/conda/envs/dyto/bin/pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   /project/rhu/dpalfaro/conda/envs/dyto/bin/pip install -e /project/rhu/dpalfaro/code/DYTO
   /project/rhu/dpalfaro/conda/envs/dyto/bin/pip install finch-clust==0.2.0 decord

4. Cache model (sets HF_HOME before offline job):
   export HF_HOME=/project/rhu/dpalfaro/cache/huggingface
   # weights downloaded in step 2 are already local; no extra caching needed.

Notes:
- DyTo provides its own dyto.llava package — do NOT put HoliTom/LLaVA-NeXT on PYTHONPATH.
- finch-clust (not just scikit-learn) is required for FINCH clustering inside DyTo.
- RoPE scaling factor 2 is mandatory for Llama-2 to handle 100-frame token counts.
- temporal_aggregation kwarg is intercepted in dyto/llava/model/llava_arch.py.
"""

import argparse
import json
import logging
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VIDEO_BASE  = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Video loading — subprocess-isolated for NFS stale-handle safety
# ---------------------------------------------------------------------------
def load_video_frames(video_path: str, num_frames: int):
    """
    Returns (pil_frames, image_sizes) matching DyTo's load_video() output format.
    image_sizes: list of (width, height) tuples, one per frame.
    """
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0))
            total = len(vr)
            idx = np.linspace(0, total - 1, n, dtype=np.int64).tolist()
            frames = vr.get_batch(idx).asnumpy()
            q.put(("ok", frames))
        except Exception as e:
            q.put(("error", str(e)))

    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frames, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill()
        proc.join(timeout=5)
        LOGGER.warning("load_video_frames: NFS timeout, returning black frames: %s", video_path)
        black = np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
        pil = [Image.fromarray(f) for f in black]
        return pil, [pil[0].size] * num_frames
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == "error":
        LOGGER.warning("load_video_frames: decode error, returning black frames: %s — %s",
                       video_path, data)
        black = np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
        pil = [Image.fromarray(f) for f in black]
        return pil, [pil[0].size] * num_frames

    pil_frames = [Image.fromarray(f) for f in data]
    image_sizes = [img.size for img in pil_frames]  # (width, height) per PIL convention
    return pil_frames, image_sizes


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, rope_scaling_factor: int = 2):
    from dyto.llava.model.builder import load_pretrained_model
    from dyto.llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=model_name,
        rope_scaling_factor=rope_scaling_factor,
    )
    model = model.cuda()
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, pil_frames, image_sizes,
                  question: str, conv_template: str,
                  temporal_aggregation: str) -> str:
    from dyto.llava.mm_utils import tokenizer_image_token, process_images
    from dyto.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from dyto.llava.conversation import conv_templates

    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    image_tensor = process_images(pil_frames, image_processor, model.config)
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=16,
        use_cache=True,
        temporal_aggregation=temporal_aggregation,
    )

    # Vicuna output includes full prompt — split on ASSISTANT:
    full_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if "ASSISTANT:" in full_text:
        return full_text.split("ASSISTANT:")[-1].strip()
    return full_text


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def find_video(video_path: str) -> str | None:
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full):
            return full
    return None


def score_prediction(prediction: str, ground_truth: str):
    gt = ground_truth.strip().upper()
    if gt == "NA":
        return None
    match = re.search(r"\b([A-D])\b", prediction.upper())
    pred = match.group(1) if match else prediction.strip().upper()[:1]
    return int(pred == gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",     required=True,
                        help="Path to llava-v1.6-vicuna-7b weights")
    parser.add_argument("--meta-path",      required=True,
                        help="MotionBench JSONL metadata file")
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--conv-template",  default="image_seq_v3",
                        help="DyTo conversation template (default: image_seq_v3)")
    parser.add_argument("--num-frames",     type=int, default=100,
                        help="Frames extracted from video before DyTo selection (default: 100)")
    parser.add_argument("--temporal-aggregation", default="spatial_tome_finch_dynamic_all_frms",
                        help="DyTo temporal aggregation strategy")
    parser.add_argument("--rope-scaling",   type=int, default=2,
                        help="RoPE scaling factor for Llama-2 context extension (default: 2)")
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument("--resume",         action="store_true",
                        help="Skip samples already written to results.jsonl")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "results.jsonl")

    done = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                r = json.loads(line)
                done.add(str(r.get("idx", "")))
        LOGGER.info("Resuming: %d samples already done", len(done))

    LOGGER.info("Loading model from %s ...", args.model_path)
    tokenizer, model, image_processor = load_model(
        args.model_path, rope_scaling_factor=args.rope_scaling
    )

    samples = []
    with open(args.meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    LOGGER.info("Evaluating %d samples (num_frames=%d, temporal_aggregation=%s)",
                len(samples), args.num_frames, args.temporal_aggregation)

    results = []
    scores  = []

    with open(out_file, "a") as fout:
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            if str(i) in done:
                continue

            video_rel  = sample["video_path"]
            question   = sample["qa"][0]["question"]
            gt         = sample["qa"][0]["answer"]
            q_type     = sample.get("question_type", "Unknown")
            video_path = find_video(video_rel)

            prediction = ""
            if video_path is None:
                LOGGER.warning("Video not found: %s", video_rel)
            else:
                try:
                    pil_frames, image_sizes = load_video_frames(video_path, args.num_frames)
                    prediction = run_inference(
                        tokenizer, model, image_processor,
                        pil_frames, image_sizes,
                        question, args.conv_template,
                        args.temporal_aggregation,
                    )
                except Exception as e:
                    LOGGER.warning("Sample %d (%s): %s", i, video_rel, e)
                    torch.cuda.empty_cache()

            s = score_prediction(prediction, gt)
            rec = {
                "idx":           i,
                "video_path":    video_rel,
                "question_type": q_type,
                "ground_truth":  gt,
                "prediction":    prediction,
                "correct":       s,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            results.append(rec)
            if s is not None:
                scores.append(s)

    total    = len(scores)
    correct  = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0

    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)", flush=True)

    summary = {
        "accuracy":         accuracy,
        "correct":          correct,
        "total_scoreable":  total,
        "total_na_skipped": na_count,
        "total_samples":    len(results),
        "model":            args.model_path,
        "dyto_params": {
            "num_frames":            args.num_frames,
            "temporal_aggregation":  args.temporal_aggregation,
            "rope_scaling":          args.rope_scaling,
            "conv_template":         args.conv_template,
        },
    }
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Results:  %s", out_file)
    LOGGER.info("Summary:  %s", summary_file)


if __name__ == "__main__":
    main()

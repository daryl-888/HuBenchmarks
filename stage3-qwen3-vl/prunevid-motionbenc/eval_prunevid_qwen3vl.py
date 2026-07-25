#!/usr/bin/env python3
"""
PruneVID (VTP) × MotionBench — Qwen3-VL-8B port.

PruneVID (Video Token Pruning, 2024): temporal segmentation -> spatial
clustering within each segment -> merge each cluster to one representative
token. Uses the authors' own `cluster_dpc_knn` (density-peak clustering),
loaded from the pinned upstream checkout, so the mechanism is theirs, not a
reimplementation.

Its published backbone is PLLaVA-7B (44.13%); this is a port, as is the
LLaVA-OV one (38.20%).

Qwen3VLForConditionalGeneration is a native HuggingFace model, NOT a LLaVA fork.
Uses transformers AutoProcessor + from_pretrained (no llava.model.builder).

Backbone: Qwen/Qwen3-VL-8B-Instruct
Weights: /project/rhu/dpalfaro/weights/qwen3-vl-8b
Uses: AutoProcessor for video preprocessing, model.generate() for inference

PYTHONPATH must reach $SRC_PRUNEVID for models.pllava.modeling_pllava.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# Model loading — Qwen3-VL native HuggingFace
# ---------------------------------------------------------------------------

def _load_prunevid_primitives():
    """
    Load PruneVID's VTP primitives WITHOUT importing PLLaVA.

    PruneVID implements its mechanism as standalone tensor functions in
    models/pllava/modeling_pllava.py — they operate on feature tensors and are
    not coupled to PLLaVA's model class. `cluster_dpc_knn` (density-peak
    clustering with k-NN density estimation) is the core of both the temporal
    segmentation and the spatial clustering steps.

    The file uses relative imports, so it cannot be loaded standalone; it must
    be imported with proper package context. $SRC_PRUNEVID points at the pinned
    upstream checkout (config/paths.sh).
    """
    import importlib.util

    src = os.environ.get("SRC_PRUNEVID", "/project/rhu/dpalfaro/code/PruneVid")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from models.pllava.modeling_pllava import cluster_dpc_knn
        return cluster_dpc_knn
    except Exception:
        pass

    path = os.path.join(src, "models", "pllava", "modeling_pllava.py")
    if not os.path.exists(path):
        raise RuntimeError(
            "PruneVID source not found at %s — set $SRC_PRUNEVID (config/paths.sh)"
            % path)
    spec = importlib.util.spec_from_file_location(
        "models.pllava.modeling_pllava", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cluster_dpc_knn


def apply_prunevid(model, cluster_ratio: float = 0.5,
                   temporal_segment_ratio: float = 0.25,
                   selected_layer: int = 10, num_frames: int = 32):
    """
    PruneVID (Video Token Pruning) ported to Qwen3-VL.

    Paper mechanism, reproduced with the authors' own `cluster_dpc_knn`:
      1. temporal segmentation — DPC-KNN over per-frame mean features,
         num_segments = num_frames * temporal_segment_ratio
      2. spatial clustering    — DPC-KNN over the visual tokens within each
         segment, num_clusters = tokens_in_segment * cluster_ratio
      3. merge                 — keep one representative token per cluster

    Two things differ from the LLaVA-OV port of this same method:

    * **Token positions.** LLaVA-OV needed a hardcoded `start=14` preamble
      offset. Qwen3-VL exposes `visual_pos_masks`, giving the exact visual
      positions — no offset guessing.
    * **How pruning is applied.** LLaVA-OV's patched Qwen2Model prunes through
      `PrunableDynamicCache.kv_cache` and calls every layer with
      attention_mask=None, so an additive mask there is silently discarded (that
      is what made the OV port a no-op at first). Qwen3-VL has no such cache, so
      we use the additive-mask path that the other seven Qwen3-VL ports are
      gate-verified on.

    Prints "PruneVID(Qwen3-VL) ACTIVE: ..." so engagement is checkable in the log.
    """
    import types
    import torch as _torch

    cluster_dpc_knn = _load_prunevid_primitives()
    lm = model.model.language_model
    orig_forward = lm.forward

    def prunevid_forward(self, *args, **kwargs):
        vis_mask = kwargs.get("visual_pos_masks", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else None
        if seq_len is None or seq_len <= 1 or vis_mask is None:
            return orig_forward(*args, **kwargs)

        vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
        n_vis = vis_idx.numel()
        if n_vis < 8:
            return orig_forward(*args, **kwargs)

        feats = inputs_embeds[0, vis_idx].float()            # (N, C)
        per_frame = max(1, n_vis // max(1, num_frames))
        n_seg = max(1, int(num_frames * temporal_segment_ratio))

        # --- step 1: temporal segmentation over per-frame mean features -------
        n_full = (n_vis // per_frame) * per_frame
        keep_local = []
        try:
            if n_full >= per_frame and num_frames > 1:
                frames = feats[:n_full].view(-1, per_frame, feats.shape[-1])
                frame_mean = frames.mean(1).unsqueeze(0)      # (1, F, C)
                n_seg_eff = max(1, min(n_seg, frame_mean.shape[1]))
                seg_id, _ = cluster_dpc_knn(frame_mean,
                                            cluster_num=n_seg_eff, k=3)
                seg_id = seg_id[0]                            # (F,)
            else:
                seg_id = _torch.zeros(1, dtype=_torch.long, device=feats.device)
                frames = None

            # --- step 2+3: cluster spatially WITHIN each segment, keep 1/cluster
            if frames is None:
                segments = [_torch.arange(n_vis, device=feats.device)]
            else:
                n_frames_eff = frames.shape[0]
                tok_idx = _torch.arange(n_full, device=feats.device)
                frame_of_tok = tok_idx // per_frame
                segments = []
                for s in _torch.unique(seg_id):
                    fmask = (seg_id == s)
                    sel = fmask[frame_of_tok.clamp(max=n_frames_eff - 1)]
                    segments.append(tok_idx[sel])
                # tokens past the last full frame form their own trailing segment
                if n_full < n_vis:
                    segments.append(_torch.arange(n_full, n_vis,
                                                  device=feats.device))

            for seg in segments:
                if seg.numel() == 0:
                    continue
                if seg.numel() <= 2:
                    keep_local.extend(seg.tolist())
                    continue
                sub = feats[seg].unsqueeze(0)                 # (1, M, C)
                n_clusters = max(1, int(seg.numel() * cluster_ratio))
                cid, _ = cluster_dpc_knn(sub, cluster_num=n_clusters, k=7)
                cid = cid[0]
                seen = set()
                for i in range(cid.numel()):
                    c = int(cid[i])
                    if c not in seen:
                        seen.add(c)
                        keep_local.append(int(seg[i]))
        except Exception as e:
            import logging
            logging.warning("PruneVID: clustering failed (%s) — passing through", e)
            return orig_forward(*args, **kwargs)

        if not keep_local:
            return orig_forward(*args, **kwargs)
        keep = vis_idx[_torch.tensor(sorted(set(keep_local)), device=vis_idx.device)]

        drop = _torch.zeros(seq_len, dtype=_torch.bool, device=feats.device)
        drop[vis_idx] = True
        drop[keep] = False
        add = _torch.zeros((1, 1, 1, seq_len), dtype=_torch.float32,
                           device=feats.device)
        add[0, 0, 0, drop] = _torch.finfo(_torch.float32).min

        if not getattr(self, "_pv_logged", False):
            import logging
            logging.warning("PruneVID(Qwen3-VL) ACTIVE: visual=%d kept=%d (%.1f%%) "
                            "segments=%d cluster_ratio=%.2f layer=%d",
                            n_vis, keep.numel(), 100.0 * keep.numel() / n_vis,
                            len(segments), cluster_ratio, selected_layer)
            self._pv_logged = True

        def _mask(module, a_, k_):
            kw = dict(k_)
            am = kw.get("attention_mask", None)
            # BUILD-MASK: identical to the seven gate-verified Qwen3-VL ports.
            # Under sdpa Qwen3-VL passes attention_mask=None (pure causal), so
            # adding to None silently drops the pruning mask. Also: this hook
            # fires on DECODE steps where q_len==1, and fp32 finfo.min overflows
            # bf16 to -inf -> NaN, so build in the hidden-state dtype.
            hs = kw.get("hidden_states")
            if hs is None and len(a_) > 0 and hasattr(a_[0], "dim"):
                hs = a_[0]
            if hs is None or hs.dim() != 3:
                return None
            q_len = hs.shape[1]
            kv_len = seq_len
            dt = hs.dtype
            neg = _torch.finfo(dt).min
            add_row = add.to(dt)[..., :kv_len]
            if am is None:
                if q_len == 1:
                    m = _torch.zeros((1, 1, 1, kv_len), dtype=dt, device=hs.device)
                else:
                    m = _torch.full((q_len, kv_len), neg, dtype=dt, device=hs.device)
                    m = _torch.triu(m, diagonal=kv_len - q_len + 1).unsqueeze(0).unsqueeze(0)
                am = m
            elif am.dim() != 4:
                return None
            kw["attention_mask"] = am + add_row
            if len(a_) >= 2 and hasattr(a_[1], "dim"):
                a_ = list(a_); a_[1] = kw["attention_mask"]; a_ = tuple(a_)
            return (a_, kw)

        # PruneVID prunes from `selected_layer` onward (the paper's layer-10
        # default), not from layer 0 — earlier layers still see all tokens.
        handles = [l.register_forward_pre_hook(_mask, with_kwargs=True)
                   for l in self.layers[selected_layer:]]
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()

    lm.forward = types.MethodType(prunevid_forward, lm)
    return model



def load_model(model_path: str, enable_prunevid: bool = False,
               cluster_ratio: float = 0.5,
               temporal_segment_ratio: float = 0.25,
               selected_layer: int = 10, num_frames: int = 32):
    """Load Qwen3-VL via transformers from_pretrained (no LLaVA)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if enable_prunevid:
        model = apply_prunevid(model, cluster_ratio=cluster_ratio,
                               temporal_segment_ratio=temporal_segment_ratio,
                               selected_layer=selected_layer,
                               num_frames=num_frames)
    model.eval()
    return None, model, processor  # Qwen3 uses processor, not tokenizer + image_processor separately


# ---------------------------------------------------------------------------
# Video loading — standard subprocess-isolated
# ---------------------------------------------------------------------------
def load_frames(video_path: str, num_frames: int) -> list:
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0))
            total = len(vr)
            indices = np.linspace(0, total - 1, n, dtype=int)
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
        proc.kill()
        proc.join(timeout=5)
        raise RuntimeError(f"load_frames: timeout (NFS stale?): {video_path}")
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == "error":
        raise RuntimeError(f"load_frames: decode error: {video_path} — {data}")

    from PIL import Image
    return [Image.fromarray(f) for f in data]


# ---------------------------------------------------------------------------
# Inference — Qwen3-VL native generate()
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(model, processor, frames: list, question: str,
                  num_frames: int = 32) -> str:
    """
    Qwen3-VL inference using native chat template + video preprocessing.
    Handles video via processor with temporal patch support.
    """
    # Build conversation using Qwen3's chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": question + POST_PROMPT},
            ],
        }
    ]

    # Apply chat template to build prompt
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process video + text together
    # CRITICAL: Qwen3VLVideoProcessor has do_sample_frames=True and fps=2, so by
    # default it RE-SAMPLES whatever frame list we hand it, ignoring --num_frames.
    # With no video_metadata it also warns "Defaulting to fps=24". We already
    # sampled exactly num_frames uniformly in load_video, so turn the processor's
    # own sampling OFF and let it consume our frames verbatim.
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames],
        return_tensors="pt",
        do_sample_frames=False,
    )
    # Report the frame count that ACTUALLY reaches the model, once, so a silent
    # re-sample can never go unnoticed again.
    if not getattr(run_inference, "_frames_logged", False):
        import logging
        pvg = inputs.get("pixel_values_videos", None)
        gt = inputs.get("video_grid_thw", None)
        logging.warning("Qwen3-VL FRAMES: requested=%d given=%d "
                        "pixel_values_videos=%s video_grid_thw=%s",
                        num_frames, len(frames),
                        tuple(pvg.shape) if pvg is not None else None,
                        gt.tolist() if gt is not None else None)
        run_inference._frames_logged = True

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=16,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def find_video(video_path: str):
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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--prunevid", action="store_true", help="apply PruneVID VTP")
    parser.add_argument("--cluster_ratio", type=float, default=0.5)
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25)
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3-VL model...", flush=True)
    _, model, processor = load_model(args.model_path, enable_prunevid=args.prunevid,
                                     cluster_ratio=args.cluster_ratio,
                                     temporal_segment_ratio=args.temporal_segment_ratio,
                                     selected_layer=args.selected_layer,
                                     num_frames=args.num_frames)

    samples = []
    with open(args.meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[:args.limit]
    print(f"Evaluating {len(samples)} samples", flush=True)

    results = []
    scores = []
    per_category = {}

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        video_path = find_video(sample["video_path"])
        question = sample["qa"][0]["question"]
        ground_truth = sample["qa"][0]["answer"]
        q_type = sample.get("question_type", "Unknown")

        if video_path is None:
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            prediction = ""
        else:
            try:
                frames = load_frames(video_path, args.num_frames)
                prediction = run_inference(
                    model, processor, frames, question,
                    num_frames=args.num_frames,
                )
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}",
                      file=sys.stderr)
                prediction = ""
                torch.cuda.empty_cache()

        s = score_prediction(prediction, ground_truth)
        results.append({
            "idx": i,
            "video_path": sample["video_path"],
            "question_type": q_type,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": s,
        })
        if s is not None:
            scores.append(s)
            per_category[q_type] = per_category.get(q_type, {"correct": 0, "total": 0})
            per_category[q_type]["total"] += 1
            per_category[q_type]["correct"] += s

    out_file = os.path.join(args.output_dir, "results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(scores)
    correct = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total_scoreable": total,
        "total_na_skipped": na_count,
        "total_samples": len(results),
        "prunevid_params": {"enabled": bool(args.prunevid),
                            "cluster_ratio": args.cluster_ratio,
                            "temporal_segment_ratio": args.temporal_segment_ratio,
                            "selected_layer": args.selected_layer,
                            "num_frames": args.num_frames},
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "num_frames": args.num_frames,
        "note": "PruneVID on Qwen3-VL — DPC-KNN temporal segmentation + spatial cluster merge, authors' cluster_dpc_knn",
        "per_category": per_category,
    }
    print(
        f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)",
        flush=True,
    )
    for cat in sorted(per_category.keys()):
        c = per_category[cat]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0.0
        print(f"  {cat}: {c['correct']}/{c['total']} = {acc:.4f}", flush=True)

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:  {out_file}", flush=True)
    print(f"Summary:  {summary_file}", flush=True)


if __name__ == "__main__":
    main()

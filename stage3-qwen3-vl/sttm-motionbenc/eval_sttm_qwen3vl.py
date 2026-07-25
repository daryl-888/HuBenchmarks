#!/usr/bin/env python3
"""
STTM (Spatio-Temporal Token Merging) × MotionBench — Qwen3-VL-8B port.

STTM merges visual tokens with a quadtree over each frame's token grid, using
the authors' own `get_quadtree_features` — the merge is a separable, fully
backbone-agnostic function, which is why this port is possible at all.

Unlike every other Qwen3-VL port here, STTM SHORTENS the sequence rather than
masking it, so position_ids / visual_pos_masks / deepstack_visual_embeds are all
rebuilt. See stage3-qwen3-vl/PORT_FEASIBILITY.md.

Its published backbone is LLaVA-Video-7B (53.33%); this is a port.
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

def _load_sttm_primitives():
    """
    Load STTM's quadtree merge WITHOUT importing its LLaVA fork.

    STTM ships two monkey-patches (LLaVA/Qwen2 and Qwen2-VL), and both are
    wholesale `*Model_forward` replacements — which is why this method was once
    written off as unportable. But both delegate the actual merging to
    `get_quadtree_features()` in `token_merging_utils/quadtree_interface.py`,
    which contains ZERO references to llava / transformers / Qwen (grep-verified).
    It is pure tensor math: a [T, C, H, W] video grid in, merged features plus
    node coordinates out. That function is the method; the rest is bookkeeping.

    $SRC_STTM points at the pinned upstream checkout (config/paths.sh).
    """
    src = os.environ.get("SRC_STTM", "/project/rhu/dpalfaro/code/STTM")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from token_merging_utils.quadtree_interface import get_quadtree_features
        return get_quadtree_features
    except Exception as e:
        raise RuntimeError(
            "STTM source not importable from %s (%s) — set $SRC_STTM (config/paths.sh)"
            % (src, e))


def apply_sttm(model, tree_thresh: float = 0.85, temporal_thresh: float = -1.0,
               root_level: int = 0, start_layer: int = 2,
               weighted_avg: bool = False, num_frames: int = 32):
    """
    STTM (Spatio-Temporal Token Merging) ported to Qwen3-VL.

    Paper mechanism, using the authors' own `get_quadtree_features`: build a
    quadtree over each frame's token grid, merging a node's four children when
    they are similar enough (cosine >= tree_thresh); optionally merge across
    frames (temporal_thresh > 0). Merging starts at decoder layer `start_layer`,
    so earlier layers still see every token.

    This port differs from the other eight Qwen3-VL ports in a way that matters:

    * **They mask, this one shortens.** Every previously gated port keeps the
      sequence length fixed and pushes -inf into the attention mask. STTM
      physically replaces the visual span with fewer tokens, so `position_ids`,
      `position_embeddings`, `visual_pos_masks` and `deepstack_visual_embeds`
      must all be rebuilt to the new length or the model breaks (or silently
      ignores the merge).

    * **deepstack has no Qwen2-VL precedent.** Qwen3-VL re-injects visual
      features at layers 8/16/24 via `_deepstack_process`, which does
      `hidden_states[visual_pos_masks] += visual_embeds`. That is an exact
      shape contract: if the visual span shrinks to K tokens, BOTH the mask and
      every `deepstack_visual_embeds[i]` must be reduced to those same K rows,
      selected in the same order. The quadtree's `tyxyx_tlbr[:, 0:3]` gives the
      (t, y, x) of each merged node, so `t*H*W + y*W + x` is exactly that row
      selector — the same index the Qwen2-VL patch uses for position_ids.

    Grid arithmetic is derived at runtime, not hardcoded: Qwen3-VL emits 11,664
    visual tokens for 32 frames, which is NOT 32 rows (11664/32 = 364.5).
    `temporal_patch_size=2` makes it T=16 planes of 729 = 27x27. 27 is odd, but
    the authors' `avgpool_to_even_side_feature` handles odd sides explicitly.

    Prints "STTM(Qwen3-VL) ACTIVE: ..." so engagement is checkable.
    """
    import types
    import torch as _torch
    import einops as _einops

    get_quadtree_features = _load_sttm_primitives()
    lm = model.model.language_model
    orig_forward = lm.forward

    def _grid_from(model_, n_vis):
        """Recover (T, H, W) for the visual span. Prefer the processor's own
        video_grid_thw; fall back to factoring n_vis into T x S x S."""
        g = getattr(model_, "_sttm_grid_thw", None)
        msz = int(getattr(getattr(model_, "config", None), "vision_config", None)
                  .spatial_merge_size) if hasattr(
                      getattr(model_, "config", None), "vision_config") else 2
        if g is not None:
            t, h, w = int(g[0]), int(g[1]) // msz, int(g[2]) // msz
            if t * h * w == n_vis:
                return t, h, w
        # fall back: find T such that n_vis/T is a perfect square
        import math
        for t in range(1, min(n_vis, 129)):
            if n_vis % t:
                continue
            per = n_vis // t
            s = int(math.isqrt(per))
            if s * s == per:
                return t, s, s
        return None

    def sttm_forward(self, *args, **kwargs):
        vis_mask = kwargs.get("visual_pos_masks", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        deepstack = kwargs.get("deepstack_visual_embeds", None)
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else None
        if seq_len is None or seq_len <= 1 or vis_mask is None:
            return orig_forward(*args, **kwargs)

        vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
        n_vis = vis_idx.numel()
        if n_vis < 16:
            return orig_forward(*args, **kwargs)

        # The visual span must be contiguous for a slice-and-splice rebuild.
        start, end = int(vis_idx[0]), int(vis_idx[-1]) + 1
        if end - start != n_vis:
            if not getattr(self, "_sttm_warned", False):
                import logging
                logging.warning("STTM: visual span not contiguous — passing through")
                self._sttm_warned = True
            return orig_forward(*args, **kwargs)

        grid = _grid_from(model, n_vis)
        if grid is None:
            if not getattr(self, "_sttm_warned", False):
                import logging
                logging.warning("STTM: could not factor %d visual tokens into "
                                "T x H x W — passing through", n_vis)
                self._sttm_warned = True
            return orig_forward(*args, **kwargs)
        T, H, W = grid

        visual = inputs_embeds[0, start:end]                    # (n_vis, C)
        try:
            video = _einops.rearrange(visual.float(), "(T H W) C -> T C H W",
                                      T=T, H=H, W=W)
            feats, _, tlbr = get_quadtree_features(
                video, tree_thresh, temporal_thresh, root_level,
                weighted_avg, slow_ver=False)
        except Exception as e:
            import logging
            logging.warning("STTM: quadtree failed (%s) — passing through", e)
            return orig_forward(*args, **kwargs)

        n_keep = feats.shape[0]
        if n_keep <= 0 or n_keep >= n_vis:
            # n_keep == n_vis means nothing merged: that is a no-op, and it must
            # be visible in the log rather than reported as a successful run.
            if not getattr(self, "_sttm_warned", False):
                import logging
                logging.warning("STTM: merged %d/%d tokens — NO REDUCTION, "
                                "method is a no-op at thresh=%.2f",
                                n_keep, n_vis, tree_thresh)
                self._sttm_warned = True
            return orig_forward(*args, **kwargs)

        # Row selector into the ORIGINAL visual span, same convention as the
        # authors' Qwen2-VL patch: flat index = t*H*W + y*W + x.
        sel = (tlbr[:, 0].long() * H * W + tlbr[:, 1].long() * W
               + tlbr[:, 2].long()).clamp_(0, n_vis - 1)

        merged = feats.to(inputs_embeds.dtype).unsqueeze(0)     # (1, K, C)
        new_embeds = _torch.cat([inputs_embeds[:, :start],
                                 merged,
                                 inputs_embeds[:, end:]], dim=1)
        new_len = new_embeds.shape[1]

        # visual_pos_masks must shrink to the new sequence length.
        new_mask = _torch.zeros((1, new_len), dtype=vis_mask.dtype,
                                device=vis_mask.device)
        new_mask[0, start:start + n_keep] = True

        # deepstack: hidden_states[visual_pos_masks] += visual_embeds is an exact
        # shape contract — select the SAME rows, in the same order.
        new_deepstack = None
        if deepstack is not None:
            new_deepstack = []
            for d in deepstack:
                if d is None:
                    new_deepstack.append(d); continue
                if d.dim() == 3 and d.shape[0] == 1 and d.shape[1] == n_vis:
                    new_deepstack.append(d[:, sel, :])
                elif d.dim() == 2 and d.shape[0] == n_vis:
                    new_deepstack.append(d[sel, :])
                else:                       # unexpected layout: leave untouched
                    new_deepstack.append(d)

        # position_ids: keep sys + inst spans, select the merged visual rows.
        pos = kwargs.get("position_ids", None)
        new_pos = pos
        if pos is not None and pos.dim() == 3 and pos.shape[-1] == seq_len:
            vis_pos = pos[..., start:end][..., sel]
            new_pos = _torch.cat([pos[..., :start], vis_pos, pos[..., end:]],
                                 dim=-1)
        elif pos is not None and pos.dim() == 2 and pos.shape[-1] == seq_len:
            vis_pos = pos[:, start:end][:, sel]
            new_pos = _torch.cat([pos[:, :start], vis_pos, pos[:, end:]], dim=-1)

        if not getattr(self, "_sttm_logged", False):
            import logging
            logging.warning("STTM(Qwen3-VL) ACTIVE: visual=%d merged=%d (%.1f%%) "
                            "grid=%dx%dx%d thresh=%.2f temporal=%.2f layer=%d "
                            "seq %d->%d deepstack=%s",
                            n_vis, n_keep, 100.0 * n_keep / n_vis, T, H, W,
                            tree_thresh, temporal_thresh, start_layer,
                            seq_len, new_len,
                            0 if deepstack is None else len(deepstack))
            self._sttm_logged = True

        kwargs = dict(kwargs)
        kwargs["inputs_embeds"] = new_embeds
        kwargs["visual_pos_masks"] = new_mask
        if new_deepstack is not None:
            kwargs["deepstack_visual_embeds"] = new_deepstack
        if new_pos is not None:
            kwargs["position_ids"] = new_pos
        # A stale attention_mask still sized to the OLD length would silently
        # mis-align; drop it and let create_causal_mask rebuild for new_len.
        am = kwargs.get("attention_mask", None)
        if am is not None and hasattr(am, "shape") and am.shape[-1] == seq_len:
            kwargs["attention_mask"] = None
        kwargs.pop("input_ids", None)     # embeds and ids would disagree now
        return orig_forward(**kwargs)

    lm.forward = types.MethodType(sttm_forward, lm)
    return model




def load_model(model_path: str, enable_sttm: bool = False,
               tree_thresh: float = 0.85, temporal_thresh: float = -1.0,
               root_level: int = 0, start_layer: int = 2,
               weighted_avg: bool = False, num_frames: int = 32):
    """Load Qwen3-VL via transformers from_pretrained (no LLaVA)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if enable_sttm:
        model = apply_sttm(model, tree_thresh=tree_thresh,
                           temporal_thresh=temporal_thresh,
                           root_level=root_level, start_layer=start_layer,
                           weighted_avg=weighted_avg, num_frames=num_frames)
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

    # Hand the true grid to the STTM hook: it needs (T, H, W) to rebuild the
    # token plane, and 11664 tokens do NOT imply 32 rows (temporal_patch_size=2).
    _gt = inputs.get("video_grid_thw", None)
    if _gt is not None:
        try:
            model._sttm_grid_thw = _gt[0].tolist()
        except Exception:
            pass

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
    parser.add_argument("--sttm", action="store_true", help="apply STTM quadtree merging")
    parser.add_argument("--tree_thresh", type=float, default=0.85)
    parser.add_argument("--temporal_thresh", type=float, default=-1.0)
    parser.add_argument("--root_level", type=int, default=0)
    parser.add_argument("--start_layer", type=int, default=2)
    parser.add_argument("--weighted_avg", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen3-VL model...", flush=True)
    _, model, processor = load_model(args.model_path, enable_sttm=args.sttm,
                                     tree_thresh=args.tree_thresh,
                                     temporal_thresh=args.temporal_thresh,
                                     root_level=args.root_level,
                                     start_layer=args.start_layer,
                                     weighted_avg=args.weighted_avg,
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
        "sttm_params": {"enabled": bool(args.sttm),
                        "tree_thresh": args.tree_thresh,
                        "temporal_thresh": args.temporal_thresh,
                        "root_level": args.root_level,
                        "start_layer": args.start_layer,
                        "weighted_avg": args.weighted_avg,
                        "num_frames": args.num_frames},
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "num_frames": args.num_frames,
        "note": "STTM on Qwen3-VL — quadtree spatio-temporal merge (authors' get_quadtree_features); SHORTENS the sequence, rebuilds position_ids/visual_pos_masks/deepstack",
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

#!/usr/bin/env python3
"""
Step 2 of 2: Run MotionBench QA using pre-extracted features + STTM token merging.

Requires extract_features.py (Step 1) to have already produced:
  {features_dir}/features/{vid_id}.pt
  {features_dir}/metadata/{vid_id}.pkl

STTM applies spatiotemporal token merging to the Qwen2 backbone of LLaVA-OV-7B
via replace_qwen2_with_quadtree_attn(). This patches the attention mechanism
before the model is loaded, so merging is active during LLM prefill.

Usage (from STTM repo root):
    python /path/to/sttm-motionbenc/eval_motionbench.py \
        --model_path /project/rhu/dpalfaro/weights/llava-ov-7b \
        --features_dir /project/rhu/dpalfaro/sttm_features/motionbench \
        --output_dir /project/rhu/dpalfaro/results/sttm_run1 \
        --sa_start_layer_idx 2 \
        --sa_tree_thresh 0.85 \
        --sa_tree_temporal_thresh 0.65 \
        --sa_tree_root_level 1

STTM params (LLaVA-OV-7B defaults from STTM paper / run_vidqa.sh):
    sa_start_layer_idx 2       LLM layer where quadtree merging activates
    sa_tree_thresh 0.85        spatial token similarity threshold
    sa_tree_temporal_thresh 0.65   temporal coherence threshold
    sa_tree_root_level 1       quadtree hierarchy depth
"""
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# STTM patch — must be called BEFORE loading the model
# ---------------------------------------------------------------------------
def apply_sttm_patch(args):
    """
    Apply STTM's quadtree attention monkey-patch to Qwen2Model.
    replace_qwen2_with_quadtree_attn() directly sets class attributes on
    Qwen2Model and replaces its forward() — so it must run before load_pretrained_model().
    """
    from token_merging_monkey_patch.quadtree_attn_monkey_patch import (
        replace_qwen2_with_quadtree_attn,
    )
    replace_qwen2_with_quadtree_attn(
        sa_start_layer_idx=args.sa_start_layer_idx,
        sa_tree_thresh=args.sa_tree_thresh,
        sa_tree_temporal_thresh=args.sa_tree_temporal_thresh,
        sa_tree_root_level=args.sa_tree_root_level,
    )
    print(
        f"STTM patch applied: layer={args.sa_start_layer_idx} "
        f"thresh={args.sa_tree_thresh} temporal_thresh={args.sa_tree_temporal_thresh} "
        f"root_level={args.sa_tree_root_level}"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    from llava.model.builder import load_pretrained_model

    tokenizer, model, _, _ = load_pretrained_model(
        model_path, None, "llava_qwen", device_map="auto"
    )
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference on pre-extracted features
# ---------------------------------------------------------------------------
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def run_inference(tokenizer, model, video_feats, question):
    """
    Run LLM on pre-extracted visual features.
    video_feats: tensor [1, N_tokens, D] from extract_features.py
    """
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # Pass pre-extracted features directly — LLaVA-OV supports this via
    # images=None + video_feats kwarg when using STTM's modified forward
    video_feats = video_feats.cuda().half()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_feats,  # pre-extracted features
            do_sample=False,
            temperature=0,
            max_new_tokens=16,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Scoring — mirrors utils.py NA-skip protocol
# ---------------------------------------------------------------------------
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
    parser.add_argument("--features_dir", required=True,
                        help="Output dir from extract_features.py "
                             "(contains features/ and metadata/)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)

    # STTM params
    parser.add_argument("--sa_start_layer_idx", type=int, default=2)
    parser.add_argument("--sa_tree_thresh", type=float, default=0.85)
    parser.add_argument("--sa_tree_temporal_thresh", type=float, default=0.65)
    parser.add_argument("--sa_tree_root_level", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    feat_dir = os.path.join(args.features_dir, "features")
    meta_dir = os.path.join(args.features_dir, "metadata")

    # Discover all extracted samples
    feat_files = sorted(Path(feat_dir).glob("*.pt"))
    if args.limit:
        feat_files = feat_files[: args.limit]
    print(f"Found {len(feat_files)} extracted samples in {feat_dir}")

    # Apply STTM patch BEFORE loading the model
    apply_sttm_patch(args)

    print("Loading model...")
    tokenizer, model = load_model(args.model_path)

    results = []
    scores = []

    for feat_path in tqdm(feat_files, desc="Evaluating"):
        vid_id = feat_path.stem
        meta_path = os.path.join(meta_dir, f"{vid_id}.pkl")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        feat = torch.load(feat_path, weights_only=True)["video_feats"]

        try:
            prediction = run_inference(tokenizer, model, feat, meta["question"])
        except Exception as e:
            print(f"  [WARN] {vid_id} failed: {e}", file=sys.stderr)
            prediction = ""

        score = score_prediction(prediction, meta["answer"])

        entry = {
            "vid_id": vid_id,
            "category": meta.get("question_type", "Unknown"),
            "original_path": meta.get("original_path", ""),
            "ground_truth": meta["answer"],
            "prediction": prediction,
            "correct": score,
        }
        results.append(entry)
        if score is not None:
            scores.append(score)

    # Save results
    out_file = os.path.join(args.output_dir, "results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(scores)
    correct = sum(scores)
    accuracy = correct / total if total > 0 else 0.0
    na_count = len(results) - total

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total_scoreable": total,
        "total_na_skipped": na_count,
        "total_samples": len(results),
        "sttm_params": {
            "sa_start_layer_idx": args.sa_start_layer_idx,
            "sa_tree_thresh": args.sa_tree_thresh,
            "sa_tree_temporal_thresh": args.sa_tree_temporal_thresh,
            "sa_tree_root_level": args.sa_tree_root_level,
        },
    }
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)")

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results : {out_file}")
    print(f"Summary : {summary_file}")


if __name__ == "__main__":
    main()

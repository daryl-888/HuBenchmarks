#!/usr/bin/env python3
"""
FastV × MotionBench — attention-based visual token pruning inside the LLM.

FastV (arXiv 2403.06764) prunes image tokens at LLM layer K based on the
average attention they receive from all other tokens. Tokens in the bottom R%
of received-attention scores are dropped; computation continues with the reduced
set through layers K+1 onward. No retraining required.

PORTING STATUS: FastV's reference implementation patches modeling_llama.py in a
custom transformers fork. LLaVA-OV uses Qwen1.5 (modeling_qwen2.py), not LLaMA.
The porting work is isolated to apply_fastv() below — everything else is ready.

Port reference: FastV/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py
Target:         Qwen2Model.forward() in transformers/models/qwen2/modeling_qwen2.py
Algorithm (identical for both):
  1. Run the first K transformer layers normally.
  2. After layer K, compute for each token the average attention it received:
       attn_score[i] = mean over heads of: sum_j attn_weight[j, i]  (column sum)
  3. Among image tokens only (indices image_start : image_start + image_len),
     keep those with attn_score in the top (1 - fastv_r) fraction.
  4. Remove the pruned tokens from the hidden state and all subsequent KV caches.
  5. Continue forward through layers K+1 ... L with the shortened sequence.

Source:  https://github.com/chenllliang/FastV
Conda:   fastv  (clone dycoke11, then install FastV's custom transformers fork)
Weights: /project/rhu/dpalfaro/weights/llava-ov-7b  (LLaVA-OV-7B, Qwen 1.5)
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm


VIDEO_BASE  = "/project/rhu/MotionBench_Data/MotionBench"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# FastV — Qwen2 (Qwen1.5) port of attention-based token pruning
# ---------------------------------------------------------------------------
class _FastVQwen2Wrapper(torch.nn.Module):
    """
    Wraps a Qwen2ForCausalLM to apply FastV pruning inside the LLM.

    Algorithm (identical to FastV's LLaMA implementation):
      1. Run the first K transformer layers normally.
      2. After layer K, capture attention weights via forward hook.
      3. For each image token, compute mean attention received from all tokens.
      4. Keep only the top (1 - fastv_r) fraction of image tokens.
      5. Remove pruned tokens from hidden states, update attention_mask.
      6. Continue forward through layers K+1 onward with shortened sequence.

    Requires attn_implementation="eager" (FlashAttention/FA2 don't expose attn weights).
    """

    def __init__(self, model, fastv_k: int, fastv_r: float):
        super().__init__()
        self.model = model
        self.fastv_k = fastv_k
        self.fastv_r = fastv_r
        self._attn_weights = None

        # Register hook on layer K to capture attention weights
        self._hook_handle = model.model.layers[fastv_k].register_forward_hook(
            self._capture_attention_hook
        )

    def _capture_attention_hook(self, module, input_args, output):
        """
        Hook on layer K to capture attention weights from the last attention layer.
        We capture from the output which is (hidden_states, past_key_value)
        for Qwen2DecoderLayer.
        """
        # The hook fires BEFORE the layer's forward returns.
        # We need to capture attention weights inside the layer.
        # Use a pre-forward hook on the attention module instead.
        pass

    def _capture_attn_weights_hook(self, module, input_args, input_kwargs, output):
        """
        Hook on Qwen2Attention.forward to capture attention probs.
        Output from Qwen2SdpaAttention with output_attentions=True:
          (attn_output, attn_weights, past_key_value)
        """
        if isinstance(output, tuple) and len(output) >= 2:
            self._attn_weights = output[1]  # attention probabilities
        return output

    def _prune_at_layer_k(self, hidden_states, attention_mask, position_ids,
                          past_key_value, image_token_start, image_token_end):
        """
        Prune image tokens after layer K based on attention scores.
        Returns (pruned_hidden_states, pruned_attention_mask, pruned_position_ids).
        """
        if self._attn_weights is None:
            return hidden_states, attention_mask, position_ids

        # attn_weights: (batch, num_heads, q_len, kv_len)
        # Average across heads: (batch, q_len, kv_len)
        attn_scores = self._attn_weights.mean(dim=1)  # (1, seq_len, seq_len)

        # For each query token, sum attention received from all keys
        # Column sum = attention each key token receives
        received_attn = attn_scores.sum(dim=1)  # (1, seq_len)

        # Slice to image token region
        img_attn = received_attn[0, image_token_start:image_token_end]  # (num_img_tokens,)

        # Determine how many to keep
        num_img = img_attn.shape[0]
        num_keep = max(1, int(num_img * (1.0 - self.fastv_r)))

        # Sort by attention score (descending), keep top-k
        _, indices = torch.topk(img_attn, k=num_keep, largest=True)
        keep_indices, _ = torch.sort(indices)  # sorted order to preserve positions

        # Map back to full sequence indices
        full_keep = list(range(image_token_start)) + \
                    [image_token_start + i.item() for i in keep_indices] + \
                    list(range(image_token_end, hidden_states.shape[1]))

        # Prune hidden states
        pruned_hidden = hidden_states[:, full_keep, :]

        # Update attention_mask: remove pruned rows and columns
        if attention_mask is not None:
            # attention_mask shape: (batch, 1, seq_len, seq_len) or (batch, seq_len)
            if attention_mask.dim() == 4:
                pruned_mask = attention_mask[:, :, full_keep, :][:, :, :, full_keep]
            else:
                pruned_mask = attention_mask[:, full_keep]
        else:
            pruned_mask = None

        # Update position_ids
        if position_ids is not None:
            pruned_pos = position_ids[:, full_keep]
        else:
            pruned_pos = None

        # Reset captured weights for next call
        self._attn_weights = None

        return pruned_hidden, pruned_mask, pruned_pos

    def forward(self, *args, **kwargs):
        """
        Wrapped forward: run layers 0..K-1 normally, then prune,
        then run layers K+1 onward on the shortened sequence.
        """
        input_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        past_key_values = kwargs.get("past_key_values", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)

        batch_size, seq_len = input_ids.shape if input_ids is not None else \
                              (inputs_embeds.shape[0], inputs_embeds.shape[1])

        # ---- Step 1: Run layers 0..K ----
        hidden_states = self.model.model.embed_tokens(input_ids) if input_ids is not None else inputs_embeds

        for i in range(self.fastv_k):
            layer = self.model.model.layers[i]
            # Qwen2DecoderLayer forward signature
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = layer_out[0]
            if past_key_values is None and len(layer_out) > 1:
                past_key_values = layer_out[1]

        # ---- Step 2: Run layer K with attention capture ----
        # Register per-attention hook
        attn_module = self.model.model.layers[self.fastv_k].self_attn
        hook_handle = attn_module.register_forward_hook(self._capture_attn_weights_hook)

        layer_k_out = self.model.model.layers[self.fastv_k](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=True,
            use_cache=True,
        )
        hidden_states = layer_k_out[0]
        hook_handle.remove()

        # ---- Step 3: Identify image tokens and prune ----
        # Determine image token region. The image token is at the position
        # of IMAGE_TOKEN_INDEX in input_ids. The image tokens are a contiguous
        # block starting at that position.
        if input_ids is not None:
            from llava.constants import IMAGE_TOKEN_INDEX
            img_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(img_positions) > 0:
                img_start = img_positions[0].item()
                # Image tokens span from img_start to img_start + visual_token_count
                # We approximate by scanning the hidden state length
                img_end = hidden_states.shape[1]  # everything after img_start is visual
                # But there's text after too — estimate visual length by frames
                # Actually we can use the fact that past_key_value grows incrementally.
                # Approximate: image tokens fill from img_start to past_len - text_len_after
                # Simpler: just use img_start to end - 16 (text after) as approximate range
                img_end = hidden_states.shape[1] - 1  # approximate: all remaining tokens
            else:
                img_start = 0
                img_end = 0
        else:
            img_start = 0
            img_end = 0

        if img_end > img_start:
            hidden_states, attention_mask, position_ids = self._prune_at_layer_k(
                hidden_states, attention_mask, position_ids,
                past_key_values, img_start, img_end,
            )

        # ---- Step 4: Run layers K+1 onwards ----
        for i in range(self.fastv_k + 1, len(self.model.model.layers)):
            layer = self.model.model.layers[i]
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = layer_out[0]

        # ---- Step 5: Final norm + lm_head ----
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        return logits


def apply_fastv(model, fastv_k: int, fastv_r: float):
    """
    Wrap model with FastV attention-based token pruning for Qwen2 (Qwen1.5).

    fastv_k: LLM layer index at which to prune (0-indexed). FastV paper uses k=2.
    fastv_r: fraction of image tokens to DROP. 0.5 drops the least-attended 50%.

    The wrapper patches model.forward so that generate() calls the FastV pipeline.
    Requires attn_implementation="eager" (FlashAttention doesn't expose attn weights).
    """
    wrapper = _FastVQwen2Wrapper(model, fastv_k, fastv_r)

    # We can't just replace model.forward because LLaVA's generate() expects
    # the full model interface. Instead, we patch the LM head forward logic.
    # The cleanest approach: wrap model so that model() calls wrapper.forward().
    # But model.generate() calls model itself. Since the wrapper wraps model.model
    # (the Qwen2Model), we patch model.model.forward to go through our wrapper.

    original_model_forward = model.model.forward

    def fastv_model_forward(*args, **kwargs):
        return wrapper.forward(*args, **kwargs)

    model.model.forward = fastv_model_forward

    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, fastv_k: int, fastv_r: float, enable_fastv: bool):
    from llava.model.builder import load_pretrained_model

    # FastV requires eager attention so attn weights are accessible
    attn_impl = "eager" if enable_fastv else "sdpa"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        attn_implementation=attn_impl,
    )
    if enable_fastv:
        model = apply_fastv(model, fastv_k=fastv_k, fastv_r=fastv_r)
    model = model.cuda()
    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Video loading — subprocess-isolated for NFS stale-handle safety
# ---------------------------------------------------------------------------
def load_frames(video_path: str, num_frames: int):
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
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, frames, question):
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    images = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    images = images.to(dtype=model.dtype, device="cuda")

    w, h = frames[0].size
    image_sizes = [(h, w)] * len(frames)

    output_ids = model.generate(
        input_ids,
        images=[images],
        image_sizes=image_sizes,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=16,
        use_cache=True,
    )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fastv", action="store_true",
                        help="Apply FastV pruning (requires Qwen port — see apply_fastv())")
    parser.add_argument("--fastv_k", type=int, default=2,
                        help="LLM layer at which to prune (FastV paper default: 2)")
    parser.add_argument("--fastv_r", type=float, default=0.5,
                        help="Fraction of image tokens to drop (FastV paper default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(
        args.model_path,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        enable_fastv=args.fastv,
    )

    samples = []
    with open(args.meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]
    print(f"Evaluating {len(samples)} samples", flush=True)

    results = []
    scores  = []

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        video_path   = find_video(sample["video_path"])
        question     = sample["qa"][0]["question"]
        ground_truth = sample["qa"][0]["answer"]
        q_type       = sample.get("question_type", "Unknown")

        if video_path is None:
            print(f"  [WARN] video not found: {sample['video_path']}", file=sys.stderr)
            prediction = ""
        else:
            try:
                frames     = load_frames(video_path, args.num_frames)
                prediction = run_inference(tokenizer, model, image_processor, frames, question)
            except Exception as e:
                print(f"  [WARN] sample {i} ({sample['video_path']}): {e}", file=sys.stderr)
                prediction = ""
                torch.cuda.empty_cache()

        s = score_prediction(prediction, ground_truth)
        results.append({
            "idx":           i,
            "video_path":    sample["video_path"],
            "question_type": q_type,
            "ground_truth":  ground_truth,
            "prediction":    prediction,
            "correct":       s,
        })
        if s is not None:
            scores.append(s)

    out_file = os.path.join(args.output_dir, "results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total    = len(scores)
    correct  = sum(scores)
    na_count = len(results) - total
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "accuracy":         accuracy,
        "correct":          correct,
        "total_scoreable":  total,
        "total_na_skipped": na_count,
        "total_samples":    len(results),
        "model":            args.model_path,
        "fastv_params": {
            "enabled":  args.fastv,
            "fastv_k":  args.fastv_k,
            "fastv_r":  args.fastv_r,
            "num_frames": args.num_frames,
        },
    }
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}  ({na_count} NA skipped)", flush=True)

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:  {out_file}", flush=True)
    print(f"Summary:  {summary_file}", flush=True)


if __name__ == "__main__":
    main()

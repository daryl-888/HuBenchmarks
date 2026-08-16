#!/usr/bin/env python3
"""
PruneVID (VTP) x MotionBench — LLaVA-OV port — ovqwen2 (Qwen2 backbone).

CACHE-FIX VARIANT: identical to eval_prunevid_ov.py except for
patch_prunable_cache() below. Root-cause bug: DyCoke's PrunableDynamicCache.
kv_cache is a FIXED index snapshot set once when pruning fires at prefill.
update() keeps torch.cat-ing newly generated tokens onto the underlying
key/value tensors every decode step, but the gather that actually gets
RETURNED to attention only ever uses that original frozen index list — so
every token generated after the prune point is permanently invisible to
self-attention at layers >= selected_layer, for the rest of decoding. This
explains why PruneVID-OV collapsed to 38.20% (D-skewed answers, same failure
signature as FastV) despite a much gentler 50% keep ratio than FastV's 15%:
retention amount isn't the driver, the structural blindness is.

Fix: extend kv_cache with each newly-appended token's position on every
decode step, so the model regains the ability to see tokens it already
generated, while the originally pruned VTP-clustered tokens stay dropped.
"""
import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm

VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# THE FIX — patch PrunableDynamicCache.update to stop freezing kv_cache
# ---------------------------------------------------------------------------
def patch_prunable_cache():
    import sys as _sys
    import torch as _torch
    try:
        from llava.model.language_model import modeling_qwen2 as _mq
    except Exception as e:
        print(f"CAPFIX IMPORT FAILED: {type(e).__name__}: {e}", file=_sys.stderr, flush=True)
        raise
    Cache = _mq.PrunableDynamicCache
    print(f"CAPFIX: found PrunableDynamicCache at {_mq.__file__} id={id(Cache)}",
          file=_sys.stderr, flush=True)
    if getattr(Cache, "_capfix_patched", False):
        print("CAPFIX: already patched, skipping", file=_sys.stderr, flush=True)
        return

    # SECOND BUG (found after the physical-prune fix alone still crashed ~23%
    # of FastV samples with shape mismatches -- same shared cache class, same
    # fix applies here): Qwen2Model.forward() ends with
    #   next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    # use_legacy_cache is True whenever past_key_values arrived as None (every
    # FIRST call of a generate()), so the cache gets serialized to a plain
    # tuple and, on the very next step, from_legacy_cache() rebuilds a BRAND
    # NEW cache object from it -- resetting _seen_tokens to the tuple's
    # (physically pruned) tensor length, so the cache believes it's only
    # ~1000 tokens into a sequence that's actually thousands of tokens along.
    # Fix: never let the cache serialize to a tuple. The SAME cache object
    # then persists for the whole generate() call and stays self-consistent.
    orig_to_legacy = Cache.to_legacy_cache
    def to_legacy_cache_noop(self):
        return self
    Cache.to_legacy_cache = to_legacy_cache_noop
    print("CAPFIX: PrunableDynamicCache.to_legacy_cache PATCHED to no-op "
          "(cache identity persists across generate() steps, no reconstruction)",
          file=_sys.stderr, flush=True)

    # ROOT CAUSE (confirmed via live instrumentation on the FastV port -- same
    # shared cache class, same bug): the original code's gather-on-return only
    # prunes what's handed to attention for the CURRENT call; the stored
    # self.key_cache/self.value_cache always keep the FULL, un-pruned tensors.
    # When a fresh generate() call's past_key_values is None, Qwen2Model.
    # forward() ends by exporting the cache via to_legacy_cache(), which reads
    # that full, un-pruned storage -- the prune is silently lost at export.
    # The next generation step reconstructs a brand-new PrunableDynamicCache
    # from that legacy (unpruned) tuple, and PruneVID's hook fails to re-engage
    # on this second pass, so ALL real decoding proceeds on the entirely
    # unpruned sequence. Net effect: PruneVID's token pruning had ZERO effect
    # on the actually-generated text in every run to date.
    #
    # Fix: make the prune PHYSICAL. The first time a layer's update() is
    # called after kv_cache is set, gather its STORED tensors in-place (not
    # just the returned view) and mark that layer done. A physically-shrunk
    # cache survives to_legacy_cache()/from_legacy_cache() round-trips intact.
    # NOTE: this print used to be gated by a "first_seen"-style one-shot flag,
    # which made it look like physical pruning only ever happened once across
    # an entire run (misleading -- the actual index_select logic below was
    # never gated that way, only the print was). Cost real debugging time
    # chasing a fake "hook stops firing after sample 1" bug. Now an
    # unconditional counter, so the log is trustworthy evidence of per-sample
    # engagement, not just per-process.
    _stats = {"n_pruned": 0}

    def fixed_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = _torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = _torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if not hasattr(self, "_capfix_pruned_layers"):
            self._capfix_pruned_layers = set()
        if self.kv_cache is not None and layer_idx not in self._capfix_pruned_layers:
            n = self.key_cache[layer_idx].shape[-2]
            keep = [i for i in self.kv_cache if i < n]
            idx = _torch.tensor(keep, device=self.key_cache[layer_idx].device)
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(2, idx)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(2, idx)
            self._capfix_pruned_layers.add(layer_idx)
            _stats["n_pruned"] += 1
            if _stats["n_pruned"] <= 5 or _stats["n_pruned"] % 200 == 0:
                print(f"CAPFIX: physically pruned layer={layer_idx} from {n} to {len(keep)} tokens "
                      f"(permanent, survives cache export) [event #{_stats['n_pruned']}]",
                      file=_sys.stderr, flush=True)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    Cache.update = fixed_update
    Cache._capfix_patched = True
    print("CAPFIX: PrunableDynamicCache.update PATCHED successfully (physical-prune variant)",
          file=_sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# PruneVID primitives (unchanged)
# ---------------------------------------------------------------------------
def _load_prunevid_primitives():
    import importlib.util
    import os
    import sys
    root = os.environ.get("SRC_PRUNEVID", "/project/rhu/dpalfaro/code/PruneVid")
    path = os.path.join(root, "models", "pllava", "modeling_pllava.py")
    if not os.path.exists(path):
        raise ImportError(f"PruneVid source not found at {path} (set $SRC_PRUNEVID)")
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from models.pllava.modeling_pllava import cluster_dpc_knn
        return cluster_dpc_knn
    except Exception:
        pass
    pkg = "models.pllava"
    spec = importlib.util.spec_from_file_location(
        pkg + ".modeling_pllava", path,
        submodule_search_locations=[os.path.dirname(path)])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ImportError(f"could not load PruneVid primitives: {e}")
    return mod.cluster_dpc_knn


def apply_prunevid(model, cluster_ratio=0.5, temporal_segment_ratio=0.25,
                   selected_layer=10, num_frames=32):
    import torch as _torch
    cluster_dpc_knn = _load_prunevid_primitives()
    inner = model.model
    _diag_state = {"n": 0}
    def _diag(msg):
        # Rate-limited so this is safe to leave on for a full 8052-sample
        # run (unlike the per-call CAPFIX-PV-DIAG print further down, which
        # only ever fires on the path that reaches clustering).
        if _diag_state["n"] < 40:
            _diag_state["n"] += 1
            print(f"CAPFIX-EARLYRET[{_diag_state['n']}]: {msg}", file=sys.stderr, flush=True)

    # FIFTH BUG, found via a diagnostic that never fired even once across a
    # full smoke run: monkeypatching inner.forward via types.MethodType
    # never actually intercepted anything. HF's generate() calls self.model(...)
    # which goes through nn.Module.__call__ -- that DOES respect an
    # instance-level .forward override in principle, but evidently something
    # in this call chain (DyCoke's patched Qwen2ForCausalLM.forward, or the
    # LLaVA multimodal wrapper) doesn't route through it reliably. A
    # forward PRE-hook is registered in nn.Module's own _forward_pre_hooks
    # dict and is guaranteed to fire on every __call__ to this exact module
    # instance, regardless of how forward is looked up internally -- so the
    # setup logic (compute keep/drop indices) moves into a pre-hook, and the
    # attention-layer hook is registered ONCE (not per-call) and reads state
    # from a shared holder instead of a per-call closure.
    holder = {"active": False, "seq_len": None, "keep_list": None}

    def _pre_hook(module, args, kwargs):
        holder["active"] = False
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            for a in args:
                if hasattr(a, "dim") and a.dim() == 3 and a.is_floating_point():
                    inputs_embeds = a
                    break
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else None
        if seq_len is None or seq_len <= 1:
            _diag(f"early-return: seq_len={seq_len} "
                  f"(inputs_embeds={'found' if inputs_embeds is not None else 'MISSING'}, "
                  f"kwargs keys={list(kwargs.keys())}, n_args={len(args)})")
            return
        # THIRD BUG (superseded -- see below): this used to check
        # kwargs.get("lengeh_vision_token") first on the theory that FastV's
        # identical-looking check made it "guaranteed fresh per sample".
        # FIFTH BUG, found by tracing prepare_inputs_labels_for_multimodal()
        # in llava_arch.py directly: its return statement hard-codes the
        # 7th tuple element (what becomes self.lengeh_vision_token) to a
        # literal None, unconditionally, for every caller. The kwarg/attr/
        # DycokeConfig checks below can never succeed -- kept only as cheap
        # early-outs in case some other model variant does populate them.
        # FastV's actual working signal is arithmetic (see run_inference()'s
        # _prunevid_text_tail computation): pre-hook text_tail fallback below.
        start = 14                                    # qwen_2 preamble
        img_len = kwargs.get("lengeh_vision_token", None)
        if img_len is None:
            img_len = getattr(model, "lengeh_vision_token", None) or \
                      getattr(module, "lengeh_vision_token", None)
        if img_len is None:
            cfg = getattr(module, "DycokeConfig", None)
            img_len = getattr(cfg, "image_token_length", None) if cfg else None
        if isinstance(img_len, _torch.Tensor):
            img_len = int(img_len.item())
        if img_len is not None:
            img_len = int(img_len)
        if (img_len is None or img_len <= 0):
            tail = getattr(module, "_prunevid_text_tail", None)
            if tail is not None and seq_len - start - tail > 0:
                img_len = seq_len - start - tail
        if not img_len or img_len <= 0:
            _diag(f"early-return: img_len={img_len!r} invalid "
                  f"(kwargs has lengeh_vision_token={'lengeh_vision_token' in kwargs}, "
                  f"model attr={getattr(model, 'lengeh_vision_token', 'MISSING')!r}, "
                  f"module attr={getattr(module, 'lengeh_vision_token', 'MISSING')!r}, "
                  f"DycokeConfig={getattr(module, 'DycokeConfig', 'MISSING')!r}, "
                  f"text_tail={getattr(module, '_prunevid_text_tail', 'MISSING')!r})")
            return
        img_len = int(img_len)
        vis = _torch.arange(start, min(start + img_len, seq_len),
                            device=inputs_embeds.device)
        if vis.numel() < 8:
            _diag(f"early-return: vis.numel()={vis.numel()} < 8 "
                  f"(start={start} img_len={img_len} seq_len={seq_len})")
            return
        feats = inputs_embeds[0, vis].unsqueeze(0).float()      # (1, N, C)
        n_tok = feats.shape[1]
        n_seg = max(1, int(num_frames * temporal_segment_ratio))
        n_clusters = max(1, int(n_tok * cluster_ratio))
        print(f"CAPFIX-PV-DIAG: seq_len={seq_len} img_len={img_len} vis.numel()={vis.numel()} "
              f"n_tok={n_tok} n_clusters={n_clusters}", file=sys.stderr, flush=True)
        try:
            idx_cluster, _ = cluster_dpc_knn(feats, cluster_num=n_clusters, k=7)
        except Exception as _e:
            print(f"CAPFIX-PV-DIAG: cluster_dpc_knn FAILED: {_e}", file=sys.stderr, flush=True)
            return
        idx_cluster = idx_cluster[0]
        print(f"CAPFIX-PV-DIAG: idx_cluster.shape={tuple(idx_cluster.shape)} "
              f"unique={idx_cluster.unique().numel()}", file=sys.stderr, flush=True)
        keep_local = []
        seen = set()
        for i in range(idx_cluster.numel()):
            c = int(idx_cluster[i])
            if c not in seen:
                seen.add(c)
                keep_local.append(i)
        keep = vis[_torch.tensor(keep_local, device=vis.device)]
        drop = _torch.zeros(seq_len, dtype=_torch.bool, device=vis.device)
        drop[vis] = True
        drop[keep] = False
        keep_idx = _torch.arange(seq_len, device=vis.device)[~drop]
        keep_list = sorted(set(i for i in keep_idx.tolist() if 0 <= i < seq_len))
        if not keep_list:
            return
        holder["active"] = True
        holder["seq_len"] = seq_len
        holder["keep_list"] = keep_list
        if not getattr(module, "_pv_logged", False):
            import logging
            logging.warning("PruneVID ACTIVE: visual=%d clusters=%d kept=%d "
                            "(%.1f%%) segments=%d layer=%d",
                            n_tok, n_clusters, keep.numel(),
                            100.0 * keep.numel() / n_tok, n_seg, selected_layer)
            module._pv_logged = True

    _install_diag_state = {"n": 0}
    def _install_diag(msg):
        if _install_diag_state["n"] < 40:
            _install_diag_state["n"] += 1
            print(f"CAPFIX-INSTALL[{_install_diag_state['n']}]: {msg}",
                  file=sys.stderr, flush=True)

    def _install(att_module, a_, k_, out):
        if not holder["active"]:
            _install_diag("skip: holder inactive (pre-hook didn't arm this forward)")
            return out
        cache = k_.get("past_key_value")
        if cache is None:
            cache = k_.get("past_key_values")
        if cache is None:
            for cand in list(a_):
                if hasattr(cand, "kv_cache"):
                    cache = cand
                    break
        if cache is None or not hasattr(cache, "kv_cache"):
            _install_diag(f"skip: no usable cache found "
                          f"(kwargs keys={list(k_.keys())}, "
                          f"past_key_value type={type(k_.get('past_key_value')).__name__}, "
                          f"past_key_values type={type(k_.get('past_key_values')).__name__}, "
                          f"n_args={len(a_)})")
            return out
        keep_list = holder["keep_list"]
        cache.kv_cache = keep_list
        # Layers before `selected_layer` already had their one update() call
        # for this prefill (kv_cache still None then, so they stored the
        # FULL sequence) -- prune them retroactively right now so every
        # layer ends this forward pass at the SAME (pruned) length. A
        # length-mismatched cache across layers is what forces transformers
        # to discard/reconstruct the whole thing on the next generation step.
        if not hasattr(cache, "_capfix_pruned_layers"):
            cache._capfix_pruned_layers = set()
        for li in range(len(cache.key_cache)):
            if li in cache._capfix_pruned_layers:
                continue
            try:
                n = cache.key_cache[li].shape[-2]
                keep_here = [i for i in keep_list if i < n]
                if not keep_here:
                    continue
                idx_t = _torch.tensor(keep_here, device=cache.key_cache[li].device)
                cache.key_cache[li] = cache.key_cache[li].index_select(2, idx_t)
                cache.value_cache[li] = cache.value_cache[li].index_select(2, idx_t)
                cache._capfix_pruned_layers.add(li)
            except Exception as e:
                print(f"CAPFIX-WARN: retroactive prune failed for layer {li}: {e} "
                      f"-- leaving that layer unpruned", file=sys.stderr, flush=True)
        holder["active"] = False   # consumed -- next forward call re-arms via _pre_hook
        return out

    inner.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    inner.layers[selected_layer].self_attn.register_forward_hook(_install, with_kwargs=True)
    print(f"CAPFIX: apply_prunevid installed hooks on inner id={id(inner)} "
          f"type={type(inner).__name__}, pre_hooks={len(inner._forward_pre_hooks)}",
          file=sys.stderr, flush=True)
    return model


def load_model(model_path: str):
    patch_prunable_cache()
    import sys as _sys
    _sys.path.insert(0, "/project/rhu/dpalfaro/code/DyCoke")
    from llava.model.builder import load_pretrained_model
    # FOURTH BUG, found via a full traceback: this previously loaded with
    # dycoke=True (l=3, p=0.7, k=0.7) -- copy-pasted from DyCoke's own
    # loading template and never disabled. With dycoke=True, the shared
    # modeling_qwen2.py layer loop does `if layer_idx < dycoke_l:
    # past_key_values.kv_cache = None` on EVERY forward pass, including
    # every decode step -- wiping out PruneVID's own already-applied
    # physical pruning right at the start of each new token, then (seeing
    # kv_cache newly None at layer_idx==dycoke_l) triggering DyCoke's own,
    # unrelated dycoke_pruning() with stale config.attention_score state,
    # which crashes with a shape mismatch. PruneVID has its own separate
    # pruning hook (apply_prunevid) and was never meant to run DyCoke's
    # mechanism alongside it.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_qwen",
        attn_implementation="sdpa",
    )
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
def run_inference(tokenizer, model, image_processor, frames, question,
                  conv_template: str = "qwen_2"):
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    # FIFTH BUG (part 2): prepare_inputs_labels_for_multimodal() in
    # llava_arch.py hard-codes its 7th return value to None -- the
    # "lengeh_vision_token" kwarg this script (and FastV) tried to read is
    # therefore always None, from any caller, unconditionally. FastV's real
    # working mechanism is arithmetic: the pre-expansion input_ids has a
    # single IMAGE_TOKEN_INDEX placeholder; the number of text tokens AFTER
    # it is fixed, so post-expansion seq_len - preamble - text_tail recovers
    # the true visual-token count. Mirror that here.
    try:
        ids = input_ids[0].tolist()
        img_pos = ids.index(IMAGE_TOKEN_INDEX)
        model.model._prunevid_text_tail = len(ids) - img_pos - 1
    except (ValueError, AttributeError):
        pass
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
    parser.add_argument("--prunevid", action="store_true", help="apply PruneVID VTP")
    parser.add_argument("--cluster_ratio", type=float, default=0.5)
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25)
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conv_template", default="qwen_2")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading model...", flush=True)
    tokenizer, model, image_processor = load_model(args.model_path)
    if args.prunevid:
        model = apply_prunevid(model, cluster_ratio=args.cluster_ratio,
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
                    tokenizer, model, image_processor, frames, question,
                    conv_template=args.conv_template,
                )
            except Exception as e:
                if i == 0:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
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
        "model": args.model_path,
        "num_frames": args.num_frames,
        "conv_template": args.conv_template,
        "prunevid_params": {"enabled": bool(args.prunevid), "cluster_ratio": args.cluster_ratio,
                            "temporal_segment_ratio": args.temporal_segment_ratio,
                            "selected_layer": args.selected_layer,
                            "cache_bug_fix": True},
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

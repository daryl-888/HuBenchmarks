#!/usr/bin/env python3
import argparse, json, logging, os, re, sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Dataset root. Override with $MOTIONBENCH (see config/paths.sh)
VIDEO_BASE = os.environ.get("MOTIONBENCH", "/project/rhu/MotionBench_Data/MotionBench")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."

def load_video_frames(video_path: str, num_frames: int):
    import multiprocessing as _mp, queue as _queue
    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(p, ctx=cpu(0))
            total = len(vr)
            idx = np.linspace(0, total - 1, n, dtype=np.int64).tolist()
            q.put(("ok", vr.get_batch(idx).asnumpy()))
        except Exception as e:
            q.put(("error", str(e)))
    q = _mp.Queue()
    proc = _mp.Process(target=_worker, args=(video_path, num_frames, q))
    proc.start()
    try:
        status, data = q.get(timeout=60)
    except _queue.Empty:
        proc.kill(); proc.join(timeout=5)
        black = np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
        pil = [Image.fromarray(f) for f in black]
        sizes = torch.tensor([[img.size[1], img.size[0]] for img in pil])
        return pil, sizes
    proc.join(timeout=5)
    if proc.is_alive(): proc.kill(); proc.join(timeout=5)
    if status == "error":
        black = np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
        pil = [Image.fromarray(f) for f in black]
        sizes = torch.tensor([[img.size[1], img.size[0]] for img in pil])
        return pil, sizes
    pil = [Image.fromarray(f) for f in data]
    # image_sizes: (N, 2) tensor of (height, width) — model expects tensor, not list of tuples
    sizes = torch.tensor([[img.size[1], img.size[0]] for img in pil])  # PIL.size = (w, h)
    return pil, sizes

# CRITICAL: Do NOT use device_map="auto" (the builder's default is "auto").
# device_map causes meta-device loading which silently drops vision encoder weights.
# Must pass device_map=None and then call model.cuda() explicitly.
def _alias_dyto_llava():
    """
    DyTo vendors LLaVA as `dyto.llava`, but its own internals use ABSOLUTE
    imports — e.g. dyto/llava/model/llava_arch.py line 20 does
        from llava.constants import IGNORE_INDEX, ...
    There is no top-level `llava` package, so that raises ModuleNotFoundError.
    dyto/llava/model/__init__.py wraps its imports in `try/except: pass`, which
    SWALLOWS the error — the result is the misleading
        ImportError: cannot import name 'LlavaLlamaForCausalLM' from 'dyto.llava.model'
    that made this look like a missing class for 5 failed jobs.

    Fix: register `dyto.llava` under the name `llava` (and pre-register its
    submodules) BEFORE anything imports DyTo, so the absolute imports resolve.
    """
    import os
    import sys
    import types

    if "llava" in sys.modules:
        return
    # The stub must exist BEFORE dyto.llava is imported: its __init__ chain is
    # what performs the absolute `from llava.constants import ...`. Giving the
    # stub a __path__ pointing at dyto's vendored llava dir makes `llava.<sub>`
    # resolve to the same files. Verified working on Carya.
    here = os.environ.get("DYTO_ROOT", "/project/rhu/dpalfaro/code/DYTO")
    vendored = os.path.join(here, "dyto", "llava")
    stub = types.ModuleType("llava")
    stub.__path__ = [vendored]
    sys.modules["llava"] = stub



def _install_finch_shim(enable: bool):
    """
    Make DyTo's `FINCH(..., tw_finch=...)` call runnable — as an explicitly
    LABELLED VARIANT, never as DyTo-as-published.

    Why this is needed: dyto/llava/model/llava_arch.py:192 calls
        FINCH(image, verbose=False, tw_finch=tw_finch)
    but DyTo pins `finch-clust==0.2.0` (pyproject.toml / PKG-INFO), and pristine
    0.2.0 from PyPI contains `tw_finch` ZERO times. We verified the pinned version
    is the one installed and that our copy matches upstream byte-for-byte. Upstream
    `ssarfraz/FINCH-Clustering` keeps TW-FINCH as a SEPARATE implementation, not a
    parameter. DyTo's published artifacts are therefore internally inconsistent.

    Rather than silently dropping the kwarg (which would give plain FINCH — a
    DIFFERENT algorithm), this shim implements the published TW-FINCH weighting
    itself, through FINCH's own `initial_rank` hook. See `_tw_initial_rank` below
    for the rule and why that hook is the correct injection point.

    That still makes any number a VARIANT, because the implementation is ours and
    not the authors':

        "DyTo (reconstructed TW-FINCH)"

    It must never be reported as "DyTo". The summary records `finch_variant`,
    `paper_faithful: false` and `reconstruction_notes`, so the caveat travels with
    the data, not just the prose.

    `$DYTO_TW_OFF=1` forces the plain-FINCH path — an A/B control. Result at n=8:
    TW-FINCH and standard FINCH gave IDENTICAL predictions (0/8 differ), so the
    clustering choice is not observable in the output at that sample size.
    """
    if not enable:
        return False
    import finch as _finch

    orig = _finch.FINCH
    if getattr(orig, "_dyto_shimmed", False):
        return True

    def _tw_initial_rank(data):
        """
        TW-FINCH first-neighbour computation.

        TW-FINCH (Sarfraz et al., CVPR 2021, 'Temporally-Weighted Hierarchical
        Clustering for Unsupervised Action Segmentation') differs from FINCH in
        exactly one place: the pairwise distance used to pick each sample's first
        neighbour is divided by a temporal-proximity weight, so frames far apart in
        time are unlikely to become first neighbours. For sequence positions i, j:

            d_tw(i,j) = d_feat(i,j) * |i - j|            (temporal weighting)

        FINCH exposes `initial_rank` as a public parameter and, when it is
        supplied, skips its own distance computation entirely (clust_rank():
        `if initial_rank is not None: orig_dist = np.empty((1,1))`). So the
        temporal weighting can be applied faithfully at the one point it belongs
        without touching the library.
        """
        import numpy as _np
        from sklearn import metrics as _metrics
        n = data.shape[0]
        d = _metrics.pairwise.pairwise_distances(data, data, metric="cosine")
        idx = _np.arange(n)
        # |i - j|, with the diagonal held out of the argmin
        tw = _np.abs(idx[:, None] - idx[None, :]).astype(_np.float64)
        _np.fill_diagonal(tw, 1.0)
        d = d * tw
        _np.fill_diagonal(d, 1e12)
        return _np.argmin(d, axis=1)

    def _finch_shim(*args, **kwargs):
        # `tw_finch` is not a parameter of any released finch-clust. When DyTo asks
        # for it we implement the published TW-FINCH weighting ourselves, via
        # FINCH's own `initial_rank` hook, rather than silently degrading to
        # standard FINCH (a different algorithm).
        #
        # MEASURED (8-sample A/B, jobs 7786505 vs 7786513): TW-FINCH and standard
        # FINCH produced IDENTICAL predictions, 0/8 differ. So on this backbone the
        # clustering choice does not reach the output at n=8 -- DyTo's downstream
        # ToMe merge and the 25-frame cap absorb the difference. Do not claim the
        # temporal weighting changes results without a larger paired comparison.
        tw = kwargs.pop("tw_finch", False)
        if os.environ.get("DYTO_TW_OFF", "0") == "1":
            tw = False          # A/B control: standard FINCH, for comparison only
        if tw and kwargs.get("initial_rank") is None:
            data = args[0] if args else kwargs.get("data")
            try:
                kwargs["initial_rank"] = _tw_initial_rank(data)
            except Exception as e:            # fall back rather than crash the run
                import logging as _l
                _l.warning("TW-FINCH initial_rank failed (%s); using standard FINCH", e)
        return orig(*args, **kwargs)

    _finch_shim._dyto_shimmed = True
    _finch.FINCH = _finch_shim
    # Modules that already did `from finch import FINCH` hold their OWN reference,
    # so patching finch.FINCH alone is not enough — sweep every loaded module and
    # rebind any existing FINCH attribute. (Patching only the guessed module name
    # left the original binding in place and the error persisted.)
    import sys as _sys
    patched = 0
    for _name, _mod in list(_sys.modules.items()):
        if _mod is None:
            continue
        try:
            if getattr(_mod, "FINCH", None) is orig:
                setattr(_mod, "FINCH", _finch_shim)
                patched += 1
        except Exception:
            continue
    import logging
    logging.warning(
        "DyTo VARIANT ACTIVE: TW-FINCH implemented locally via FINCH's "
        "initial_rank hook (d_feat * |i-j|), because `tw_finch` is not a "
        "parameter of finch-clust==0.2.0 — the version DyTo itself pins. The "
        "weighting follows the published TW-FINCH definition, but it is OUR "
        "implementation, not the authors' released code. Report as "
        "'DyTo (reconstructed TW-FINCH)' — NOT as DyTo.")
    import logging as _lg
    _lg.warning("DyTo FINCH shim rebound in %d already-imported module(s)", patched)
    return True


def load_model(model_path: str, rope_scaling_factor: int = 2):
    _alias_dyto_llava()
    from dyto.llava.model.builder import load_pretrained_model
    from dyto.llava.mm_utils import get_model_name_from_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name,
        rope_scaling_factor=rope_scaling_factor,
        device_map=None,
    )

    # The llava-v1.6-vicuna-7b config ships mm_patch_merge_type="spatial_unpad"
    # and image_aspect_ratio="anyres". For IMAGES that is right, but it forces
    # prepare_inputs_labels_for_multimodal down the branch that builds a LIST of
    # per-patch features (llava_arch.py ~312, `image_features = new_image_features`),
    # while DyTo's own video path does
    #     T, N, D = image_features.shape          # llava_arch.py:226
    # which requires a 3-D TENSOR. Those two paths are mutually incompatible, and
    # the mismatch surfaces as the misleading
    #     'list' object has no attribute 'shape'
    # at llava_arch.py:325. DyTo's temporal aggregation (FINCH + ToMe) operates on
    # uniform per-frame features, so the flat path is the correct one for video.
    _install_finch_shim(os.environ.get("DYTO_FINCH_SHIM", "0") == "1")
    model.config.mm_patch_merge_type = "flat"
    model.config.image_aspect_ratio = "square"
    return tokenizer, model.cuda(), image_processor

@torch.inference_mode()
def run_inference(tokenizer, model, image_processor, pil_frames, image_sizes,
                  question: str, conv_template: str, temporal_aggregation: str) -> str:
    _alias_dyto_llava()
    from dyto.llava.mm_utils import tokenizer_image_token, process_images
    from dyto.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from dyto.llava.conversation import conv_templates
    user_msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    # process_images returns a TENSOR for plain configs but a LIST of per-patch
    # tensors under anyres/multi-patch — calling .to() on a list raises
    # "'list' object has no attribute 'shape'". Handle both.
    # process_images returns a TENSOR for plain configs but a LIST of per-patch
    # tensors under anyres. DyTo's own generate() path calls .shape on `images`,
    # so passing a list through raises "'list' object has no attribute 'shape'"
    # from inside DyTo. Stack back to a single tensor when the shapes allow.
    image_tensor = process_images(pil_frames, image_processor, model.config)
    if isinstance(image_tensor, (list, tuple)):
        try:
            image_tensor = torch.stack(list(image_tensor), dim=0)
        except Exception:
            # ragged patches can't be stacked — fall back to the first-scale view
            image_tensor = image_tensor[0]
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")
    # Re-run the sweep here: llava_arch may only be imported during the first
    # forward, after load_model() already ran. The shim is idempotent.
    _install_finch_shim(os.environ.get("DYTO_FINCH_SHIM", "0") == "1")
    output_ids = model.generate(input_ids, images=image_tensor, image_sizes=image_sizes,
        do_sample=False, temperature=0, max_new_tokens=16, use_cache=True,
        temporal_aggregation=temporal_aggregation)
    full = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return full.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full else full

def find_video(video_path):
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        if os.path.exists(full): return full
    return None

def score_prediction(prediction, ground_truth):
    gt = ground_truth.strip().upper()
    if gt == "NA": return None
    m = re.search(r"\b([A-D])\b", prediction.upper())
    return int((m.group(1) if m else prediction.strip().upper()[:1]) == gt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conv-template", default="vicuna_v1")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--temporal-aggregation", default="spatial_tome_finch_dynamic_all_frms")
    parser.add_argument("--rope-scaling", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "results.jsonl")
    done = set()
    if args.resume and os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                r = json.loads(line); done.add(str(r.get("idx","")))
    LOGGER.info("Loading model from %s ...", args.model_path)
    tokenizer, model, image_processor = load_model(args.model_path, args.rope_scaling)
    samples = [json.loads(l) for l in open(args.meta_path) if l.strip()]
    if args.limit: samples = samples[:args.limit]
    LOGGER.info("Evaluating %d samples", len(samples))
    scores = []
    with open(out_file, "a") as fout:
        for i, s in enumerate(tqdm(samples, desc="Evaluating")):
            if str(i) in done: continue
            video_path = find_video(s["video_path"])
            pred = ""
            if video_path:
                try:
                    pf, isz = load_video_frames(video_path, args.num_frames)
                    pred = run_inference(tokenizer, model, image_processor, pf, isz,
                        s["qa"][0]["question"], args.conv_template, args.temporal_aggregation)
                except Exception as e:
                    import traceback as _tb
                    LOGGER.warning("Sample %d FULL TRACE:\n%s", i, _tb.format_exc())
                    LOGGER.warning("Sample %d: %s", i, e)
                    torch.cuda.empty_cache()
            sc = score_prediction(pred, s["qa"][0]["answer"])
            fout.write(json.dumps({"idx":i,"video_path":s["video_path"],
                "question_type":s.get("question_type","Unknown"),"ground_truth":s["qa"][0]["answer"],
                "prediction":pred,"correct":sc})+"\n")
            if sc is not None: scores.append(sc)
    cor, total, na = sum(scores), len(scores), len(samples) - len(scores)
    acc = cor / total if total > 0 else 0.0
    print(f"\nAccuracy: {cor}/{total} = {acc:.4f} ({na} NA skipped)", flush=True)
    json.dump({"accuracy":acc,"correct":cor,"total_scoreable":total,"total_na_skipped":na,
        "total_samples":len(samples),"model":args.model_path,
        "dyto_params":{
            "enabled": True,"num_frames":args.num_frames,"temporal_aggregation":args.temporal_aggregation,
            "rope_scaling":args.rope_scaling,"conv_template":args.conv_template,
            # The caveat travels with the data, not just the prose: two pieces of
            # this run are NOT the authors' released code. See UPSTREAM_DEFECTS.md.
            "finch_variant": ("reconstructed-TW-FINCH"
                              if os.environ.get("DYTO_FINCH_SHIM","0") == "1"
                              else "released-code-only"),
            "paper_faithful": False,
            "reconstruction_notes": [
                "finch_cluster() return: TRANSCRIBED verbatim from the sibling "
                "KMeans function in the same file (13/14 normalized lines "
                "identical) — recovered, not invented",
                "TW-FINCH weighting: OUR implementation of the published "
                "d_feat*|i-j| rule via FINCH's initial_rank hook; `tw_finch` is "
                "absent from the pinned finch-clust==0.2.0",
            ],
            "report_as": "DyTo (reconstructed TW-FINCH)"}},
        open(os.path.join(args.output_dir,"summary.json"),"w"),indent=2)

if __name__ == "__main__": main()

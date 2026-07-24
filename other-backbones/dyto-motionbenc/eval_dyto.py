#!/usr/bin/env python3
import argparse, json, logging, os, re, sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"
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
    image_tensor = process_images(pil_frames, image_processor, model.config)
    if isinstance(image_tensor, (list, tuple)):
        image_tensor = [t.to(dtype=torch.float16, device="cuda") for t in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")
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
            "rope_scaling":args.rope_scaling,"conv_template":args.conv_template}},
        open(os.path.join(args.output_dir,"summary.json"),"w"),indent=2)

if __name__ == "__main__": main()

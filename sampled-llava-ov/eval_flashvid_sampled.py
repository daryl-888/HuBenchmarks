#!/usr/bin/env python3
# ===========================================================================
# SAMPLED VARIANT — verbatim copy of
# stage1-llava-ov/flashvid-motionbenc/eval_flashvid_motionbench.py, with ONE
# behavioural change: generation is stochastic instead of greedy.
#
# The original is UNTOUCHED and still produces the gated greedy result.
# These numbers are NOT comparable to gated results and CANNOT pass the
# divergence gate. They exist to show the A/B/C/D answer DISTRIBUTION.
# ===========================================================================
"""
Evaluate FlashVID-wrapped LLaVA-OneVision on MotionBench.

This script is designed to be placed at the root of Fanziyang-v/FlashVID and run
inside the FlashVID environment. It produces MotionBench-compatible answer files
(uid -> A/B/C/D), raw generation logs, per-frame metrics, and an aggregate summary
for the requested frame counts.

Key design goals:
  1. Deterministic, auditable multiple-choice evaluation.
  2. Robust MotionBench metadata/video-path resolution across official and local layouts.
  3. Resume-safe execution with per-frame predictions.jsonl and answers.json.
  4. Optional sharding via torchrun/accelerate environment variables.
  5. Explicit controls for FlashVID vision-side compression and optional inner-LLM pruning.

Example:
  python evaluate_flashvid_motionbench.py \
      --meta-file /path/to/MotionBench/video_info.meta.jsonl \
      --video-roots /path/to/MotionBench \
      --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
      --frame-counts 8 20 33 40 \
      --enable-flashvid \
      --retention-ratio 0.10 \
      --output-dir outputs/motionbench_flashvid
"""

from __future__ import annotations

import argparse

# Sampling config; overwritten from CLI in main().
_SAMP = {"temperature": 0.7, "top_p": 0.9, "greedy": False}
import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm


LOGGER = logging.getLogger("flashvid_motionbench")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
CHOICES = ("A", "B", "C", "D")


# -----------------------------
# Generic utilities
# -----------------------------


def setup_logger(log_file: Optional[Path] = None, rank: int = 0) -> None:
    LOGGER.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = rank
            return True

    LOGGER.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(RankFilter())
    LOGGER.addHandler(stream)

    if log_file is not None and rank == 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.addFilter(RankFilter())
        LOGGER.addHandler(fh)


def set_reproducible(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Keep generation deterministic without forcing slower deterministic kernels.
    torch.set_grad_enabled(False)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_idx}: {exc}") from exc
    return rows


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def safe_stem(value: str) -> str:
    value = value.strip().replace("\\", "/")
    return Path(value).stem


def get_dist_info() -> Tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from torchrun/accelerate/slurm env."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(rank))))
    return rank, world_size, local_rank


def is_rank0(rank: int) -> bool:
    return rank == 0


# -----------------------------
# MotionBench metadata handling
# -----------------------------


def normalize_qa_uid(uid: Any) -> str:
    return str(uid)


def iter_qas(meta_rows: Sequence[Dict[str, Any]]) -> Iterator[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """Yield (video_index, metadata, qa) for MotionBench-like rows."""
    for video_idx, meta in enumerate(meta_rows):
        qas = meta.get("qa")
        if not isinstance(qas, list):
            # Some parquet-style exports may flatten one QA per row.
            if any(k in meta for k in ("question", "query", "answer", "uid")):
                qas = [meta]
            else:
                LOGGER.warning("Skipping metadata row %d: no qa list and no flattened QA keys", video_idx)
                continue
        for qa in qas:
            if not isinstance(qa, dict):
                LOGGER.warning("Skipping non-dict QA in row %d: %r", video_idx, qa)
                continue
            if "uid" not in qa and "uid" in meta:
                qa = dict(qa)
                qa["uid"] = meta["uid"]
            if "uid" not in qa:
                # Stable synthetic UID to avoid silent overwrites; official metadata should have uid.
                qa = dict(qa)
                qa["uid"] = hashlib.md5(json.dumps(qa, sort_keys=True).encode("utf-8")).hexdigest()
            yield video_idx, meta, qa


def get_question_type(meta: Dict[str, Any], qa: Dict[str, Any]) -> str:
    for obj in (qa, meta):
        val = obj.get("question_type") or obj.get("category") or obj.get("task") or obj.get("type")
        if isinstance(val, list):
            return "/".join(map(str, val))
        if val is not None:
            return str(val)
    return "unknown"


def get_gt_answer(qa: Dict[str, Any]) -> Optional[str]:
    ans = qa.get("answer", qa.get("label", qa.get("gt", None)))
    if ans is None:
        return None
    ans = str(ans).strip()
    # Official MotionBench answers are A/B/C/D/NA. Normalize defensively.
    m = re.search(r"\b([ABCD])\b", ans.upper())
    if m:
        return m.group(1)
    if ans.upper() == "NA":
        return "NA"
    return ans


def extract_question_text(qa: Dict[str, Any], meta: Dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "problem"):
        val = qa.get(key, meta.get(key))
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise KeyError(f"Could not find question text in QA keys={list(qa.keys())} meta keys={list(meta.keys())}")


def normalize_option_label(label: Any, default_idx: int) -> str:
    if isinstance(label, str):
        m = re.search(r"[ABCD]", label.upper())
        if m:
            return m.group(0)
    return CHOICES[default_idx]


def extract_options(qa: Dict[str, Any], meta: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return [(A, text), ...] if options are stored outside the question string."""
    candidate_keys = (
        "options",
        "choices",
        "candidates",
        "option",
        "candidate_answers",
        "answers",
    )
    for obj in (qa, meta):
        for key in candidate_keys:
            val = obj.get(key)
            if val is None:
                continue
            if isinstance(val, dict):
                options = []
                for idx, k in enumerate(sorted(val.keys())):
                    if idx >= 4:
                        break
                    label = normalize_option_label(k, idx)
                    options.append((label, str(val[k]).strip()))
                if options:
                    return sorted(options, key=lambda x: CHOICES.index(x[0]) if x[0] in CHOICES else 99)
            if isinstance(val, list):
                options = []
                for idx, item in enumerate(val[:4]):
                    label = CHOICES[idx]
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("value") or item.get("answer") or item.get("content") or str(item)
                        label = normalize_option_label(item.get("label", item.get("key", label)), idx)
                    else:
                        text = str(item)
                    options.append((label, text.strip()))
                if options:
                    return sorted(options, key=lambda x: CHOICES.index(x[0]) if x[0] in CHOICES else 99)
    return []


def question_already_has_options(question: str) -> bool:
    # Conservative: detect at least two option markers.
    markers = 0
    for c in CHOICES:
        if re.search(rf"(^|[\n\s\(]){c}[\)\.\:\-]\s+", question, flags=re.IGNORECASE):
            markers += 1
    return markers >= 2


def build_prompt(qa: Dict[str, Any], meta: Dict[str, Any], prompt_style: str = "strict") -> str:
    question = extract_question_text(qa, meta)
    options = extract_options(qa, meta)
    if options and not question_already_has_options(question):
        option_text = "\n".join([f"{label}. {text}" for label, text in options])
        question = f"{question}\n{option_text}"

    if prompt_style == "plain":
        return question
    if prompt_style == "strict":
        return (
            "You are answering a multiple-choice video question. "
            "Choose the best option from A, B, C, and D. "
            "Respond with a single uppercase letter only.\n\n"
            f"{question}\n\nAnswer:"
        )
    if prompt_style == "motionbench":
        return (
            f"{question}\n"
            "Please answer the question by selecting only one option letter from A, B, C, and D."
        )
    raise ValueError(f"Unknown prompt_style={prompt_style}")


# -----------------------------
# Video path resolution
# -----------------------------


@dataclass
class VideoIndex:
    by_name: Dict[str, Path]
    by_stem: Dict[str, Path]
    by_uuid_like: Dict[str, Path]
    roots: List[str]


UUID_LIKE_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?:_\d+_\d+)?")


def build_video_index(video_roots: Sequence[Path], disable_recursive_index: bool = False) -> VideoIndex:
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}
    by_uuid_like: Dict[str, Path] = {}
    roots = [str(p) for p in video_roots]

    if disable_recursive_index:
        return VideoIndex(by_name, by_stem, by_uuid_like, roots)

    LOGGER.info("Indexing videos under roots: %s", roots)
    start = time.time()
    n = 0
    for root in video_roots:
        if not root.exists():
            LOGGER.warning("Video root does not exist: %s", root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            n += 1
            name = path.name
            stem = path.stem
            by_name.setdefault(name, path)
            by_stem.setdefault(stem, path)
            m = UUID_LIKE_RE.search(stem)
            if m:
                by_uuid_like.setdefault(m.group(0), path)
    LOGGER.info("Indexed %d video files in %.1fs", n, time.time() - start)
    return VideoIndex(by_name, by_stem, by_uuid_like, roots)


def flatten_candidate_values(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        for key, val in obj.items():
            # Ignore QA text-heavy fields here to avoid treating question text as a path.
            if key in {"qa", "question", "query", "prompt", "answer", "options", "choices", "caption", "description"}:
                continue
            yield from flatten_candidate_values(val)
    elif isinstance(obj, list):
        for item in obj:
            yield from flatten_candidate_values(item)
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            yield s


def video_candidate_strings(meta: Dict[str, Any]) -> List[str]:
    prioritized_keys = (
        "video",
        "video_path",
        "video_name",
        "video_file",
        "filename",
        "file_name",
        "path",
        "filepath",
        "clip",
        "clip_name",
        "id",
        "video_id",
        "name",
    )
    cands: List[str] = []
    for key in prioritized_keys:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            cands.append(val.strip())
    for val in flatten_candidate_values(meta):
        if val not in cands:
            # Retain only values that look path/id-like.
            looks_video = any(ext in val.lower() for ext in VIDEO_EXTS)
            looks_uuid = UUID_LIKE_RE.search(val) is not None
            looks_clip = re.search(r"(^|/)(clip_\d+|S\d{2}A\d+I\d+|[A-Za-z0-9_-]{8,}_\d+_\d+)", val) is not None
            if looks_video or looks_uuid or looks_clip:
                cands.append(val)
    return cands


def resolve_video_path(meta: Dict[str, Any], index: VideoIndex) -> Path:
    candidates = video_candidate_strings(meta)
    tried: List[str] = []

    # Direct absolute or root-relative paths.
    for cand in candidates:
        cand_norm = cand.replace("\\", "/")
        p = Path(cand_norm)
        if p.is_absolute() and p.exists():
            return p
        for root_str in index.roots:
            root = Path(root_str)
            direct = root / cand_norm
            tried.append(str(direct))
            if direct.exists() and direct.is_file():
                return direct
            if direct.suffix == "":
                for ext in VIDEO_EXTS:
                    with_ext = direct.with_suffix(ext)
                    tried.append(str(with_ext))
                    if with_ext.exists() and with_ext.is_file():
                        return with_ext

    # Name/stem lookup through recursive index.
    for cand in candidates:
        cand_name = Path(cand).name
        cand_stem = safe_stem(cand)
        if cand_name in index.by_name:
            return index.by_name[cand_name]
        if cand_stem in index.by_stem:
            return index.by_stem[cand_stem]
        m = UUID_LIKE_RE.search(cand)
        if m and m.group(0) in index.by_uuid_like:
            return index.by_uuid_like[m.group(0)]

    # Last-resort: if metadata has an ID and official video is stored as id.mp4.
    for key in ("id", "video_id", "name"):
        val = meta.get(key)
        if val is None:
            continue
        stem = safe_stem(str(val))
        if stem in index.by_stem:
            return index.by_stem[stem]
        if stem in index.by_uuid_like:
            return index.by_uuid_like[stem]

    example = {k: meta.get(k) for k in ("video", "video_path", "video_name", "filename", "id", "video_id", "name") if k in meta}
    raise FileNotFoundError(
        "Could not resolve video path. "
        f"metadata_id_fields={example}; candidates={candidates[:20]}; tried_first={tried[:10]}; roots={index.roots}"
    )


# -----------------------------
# Video loading and model inference
# -----------------------------


def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video has zero frames")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)


def load_video_frames(video_path: Path, num_frames: int) -> np.ndarray:
    import multiprocessing as _mp
    import queue as _queue

    def _worker(p, n, q):
        try:
            import numpy as np
            from decord import VideoReader, cpu
            vr = VideoReader(str(p), ctx=cpu(0))
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
        LOGGER.warning("load_video_frames: timeout (NFS stale?), returning black frames: %s", video_path)
        return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    if status == "error":
        LOGGER.warning("load_video_frames: decode error, returning black frames: %s — %s", video_path, data)
        return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
    return data


def get_model_device(model: torch.nn.Module, fallback: str) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def load_llava_onevision(args: argparse.Namespace, device: str):
    # Imports are intentionally inside the function so --merge-only can run without model deps.
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = args.model_name or get_model_name_from_path(args.model_path)
    llava_model_args: Dict[str, Any] = {"multimodal": True}
    if args.attn_implementation:
        llava_model_args["attn_implementation"] = args.attn_implementation

    device_map = args.device_map
    if device_map == "rank_device":
        device_map = device

    LOGGER.info("Loading LLaVA-OneVision model=%s device_map=%s attn=%s", args.model_path, device_map, args.attn_implementation)
    try:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.model_path,
            None,
            model_name,
            device_map=device_map,
            **llava_model_args,
        )
    except Exception as exc:
        if args.attn_implementation == "flash_attention_2":
            LOGGER.warning("flash_attention_2 load failed (%s). Retrying with sdpa.", repr(exc))
            llava_model_args["attn_implementation"] = "sdpa"
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                args.model_path,
                None,
                model_name,
                device_map=device_map,
                **llava_model_args,
            )
        else:
            raise

    if args.enable_flashvid:
        from flashvid import flashvid

        if args.inner_llm_pruning:
            flashvid_expansion = args.expansion
            pruning_layer = args.pruning_layer
            llm_retention_ratio = args.llm_retention_ratio
        else:
            # Match your previous trace script:
            # main = before-LLM ADTS + TSTM only.
            # expansion must be 1.0, otherwise the before-LLM budget becomes retention_ratio * expansion.
            flashvid_expansion = 1.0
            pruning_layer = 10**9
            llm_retention_ratio = 1.0

        LOGGER.info(
            "Wrapping model with FlashVID: retention=%.3f alpha=%.3f temporal_threshold=%.3f "
            "segment=%s inner_llm_pruning=%s pruning_layer=%s llm_retention=%.3f",
            args.retention_ratio,
            args.alpha,
            args.temporal_threshold,
            args.do_segment,
            args.inner_llm_pruning,
            pruning_layer,
            llm_retention_ratio,
        )
        model = flashvid(
            model=model,
            retention_ratio=args.retention_ratio,
            do_segment=args.do_segment,
            segment_threshold=args.segment_threshold,
            min_segment_num=args.min_segment_num,
            complementary_segment=args.complementary_segment,
            token_selection_method=args.token_selection_method,
            alpha=args.alpha,
            temporal_threshold=args.temporal_threshold,
            expansion=flashvid_expansion,
            pruning_layer=pruning_layer,
            llm_retention_ratio=llm_retention_ratio,
        )

    model.eval()
    return tokenizer, model, image_processor, max_length


def build_llava_prompt(question_text: str, conv_template_name: str) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    conv = conv_templates[conv_template_name].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question_text}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def generate_answer(
    *,
    tokenizer,
    model,
    image_processor,
    video_frames: np.ndarray,
    prompt_text: str,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    prompt = build_llava_prompt(prompt_text, args.conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Match FlashVID playground/LMMs-Eval: preprocess all sampled frames as a single video tensor list.
    frames_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.float16)
    image_tensors = [frames_tensor]

    gen_kwargs: Dict[str, Any] = dict(
        images=image_tensors,
        modalities=["video"],
        do_sample=(not _SAMP["greedy"]),
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    if args.num_beams is not None and args.num_beams > 1:
        gen_kwargs["num_beams"] = args.num_beams
    # Only pass temperature/top_p when actually sampling — Transformers warns and
    # can mutate configs if they are supplied alongside do_sample=False.
    if not _SAMP["greedy"]:
        gen_kwargs["temperature"] = _SAMP["temperature"]
        gen_kwargs["top_p"] = _SAMP["top_p"]

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text


# -----------------------------
# Answer extraction and metrics
# -----------------------------


LETTER_PATTERNS = [
    re.compile(r"^\s*([ABCD])\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:answer\s*(?:is|:)?\s*)?\(?([ABCD])\)?[\.:\)]?\s*(?:$|\n|\s)", re.IGNORECASE),
    re.compile(r"\b(?:option|answer|choice)\s*(?:is|:)?\s*\(?([ABCD])\)?\b", re.IGNORECASE),
    re.compile(r"\(([ABCD])\)", re.IGNORECASE),
    re.compile(r"\b([ABCD])\s*[\.:\)]", re.IGNORECASE),
]


def extract_choice(raw_text: str, fallback: str = "A") -> Tuple[str, bool, str]:
    text = raw_text.strip()
    # Drop common chat prefixes without losing first letter answers.
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    for pat in LETTER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper(), True, pat.pattern

    # If the model generated a long sentence containing exactly one isolated A/B/C/D, use it.
    isolated = re.findall(r"\b([ABCD])\b", text.upper())
    unique = sorted(set(isolated))
    if len(unique) == 1:
        return unique[0], True, "unique_isolated_letter"

    return fallback.upper(), False, "fallback"


def compute_motionbench_accuracy_from_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute official-compatible and extra metrics from our prediction logs."""
    total_qa_num = 0
    total_answered_num = 0
    right_num = 0

    category_right = defaultdict(float)
    category_total = defaultdict(float)
    category_acc: Dict[str, Any] = {}

    valid_total = 0
    valid_right = 0
    parse_failures = 0
    errors = 0

    for row in predictions:
        total_qa_num += 1
        uid = row.get("uid")
        pred = row.get("pred_choice")
        gt = row.get("gt_answer")
        category = row.get("question_type", "unknown")
        if row.get("error"):
            errors += 1
        if not row.get("parse_ok", False):
            parse_failures += 1
        if pred is not None:
            total_answered_num += 1
        if gt == "NA" or gt is None:
            continue
        category_total[category] += 1
        valid_total += 1
        if pred == gt:
            category_right[category] += 1
            right_num += 1
            valid_right += 1

    for key in sorted(category_total.keys()):
        category_acc[key] = category_right[key] / category_total[key] if category_total[key] else 0.0

    category_acc.update(
        {
            # Official MotionBench script divides right_num by total_qa_num and answered count.
            "acc": float(right_num) / total_qa_num if total_qa_num else 0.0,
            "answered_acc": float(right_num) / total_answered_num if total_answered_num else 0.0,
            # Extra diagnostic: excludes answer==NA entries.
            "valid_non_na_acc": float(valid_right) / valid_total if valid_total else 0.0,
            "total_qa_num": total_qa_num,
            "total_answered_num": total_answered_num,
            "right_num": right_num,
            "valid_non_na_total": valid_total,
            "valid_non_na_right": valid_right,
            "parse_failures": parse_failures,
            "errors": errors,
        }
    )
    return category_acc


def load_existing_predictions(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    by_uid: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows, by_uid
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
            if "uid" in obj:
                by_uid[str(obj["uid"])] = obj
    return rows, by_uid


def write_answers_and_metrics(frame_dir: Path, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Last prediction for repeated uid wins. Official answer file is uid -> A/B/C/D.
    answers = {str(row["uid"]): row.get("pred_choice", "A") for row in predictions if "uid" in row}
    metrics = compute_motionbench_accuracy_from_predictions(list(answers_to_latest_rows(predictions).values()))
    atomic_write_json(frame_dir / "answers.json", answers, indent=2)
    atomic_write_json(frame_dir / "metrics.json", metrics, indent=2)
    return metrics


def answers_to_latest_rows(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in predictions:
        if "uid" in row:
            latest[str(row["uid"])] = row
    return latest


# -----------------------------
# Evaluation loop
# -----------------------------


@dataclass
class RunMetadata:
    command: str
    model_path: str
    enable_flashvid: bool
    frame_counts: List[int]
    prompt_style: str
    retention_ratio: float
    alpha: float
    temporal_threshold: float
    do_segment: bool
    segment_threshold: float
    min_segment_num: int
    complementary_segment: bool
    inner_llm_pruning: bool
    pruning_layer: int
    llm_retention_ratio: float
    token_selection_method: str
    seed: int
    rank: int
    world_size: int


def shard_video_indices(num_videos: int, rank: int, world_size: int) -> set:
    if world_size <= 1:
        return set(range(num_videos))
    return {i for i in range(num_videos) if i % world_size == rank}


def evaluate_one_frame_count(
    *,
    frame_count: int,
    meta_rows: List[Dict[str, Any]],
    video_index: VideoIndex,
    tokenizer,
    model,
    image_processor,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Optional[Dict[str, Any]]:
    frame_dir = Path(args.output_dir) / f"frames_{frame_count:02d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    shard_pred_path = frame_dir / ("predictions.jsonl" if world_size == 1 else f"predictions.rank{rank:03d}.jsonl")
    failure_path = frame_dir / ("failures.jsonl" if world_size == 1 else f"failures.rank{rank:03d}.jsonl")

    existing_rows, existing_by_uid = load_existing_predictions(shard_pred_path) if args.resume else ([], {})
    if args.overwrite and shard_pred_path.exists():
        shard_pred_path.unlink()
        existing_rows, existing_by_uid = [], {}
    if args.overwrite and failure_path.exists():
        failure_path.unlink()

    allowed_video_indices = shard_video_indices(len(meta_rows), rank, world_size)
    all_items = list(iter_qas(meta_rows))
    if args.limit is not None:
        all_items = all_items[: args.limit]
    items = [(vid_idx, meta, qa) for vid_idx, meta, qa in all_items if vid_idx in allowed_video_indices]

    LOGGER.info(
        "Frame count=%d | rank workload: %d QA items from %d videos | resume_existing=%d",
        frame_count,
        len(items),
        len({x[0] for x in items}),
        len(existing_by_uid),
    )

    if args.dry_run:
        for vid_idx, meta, qa in items[: min(5, len(items))]:
            vp = resolve_video_path(meta, video_index)
            prompt = build_prompt(qa, meta, args.prompt_style)
            LOGGER.info("DRY uid=%s video=%s prompt=%s", qa.get("uid"), vp, prompt[:500].replace("\n", " | "))
        return None

    pbar = tqdm(items, desc=f"frames={frame_count} rank={rank}", disable=rank != 0)
    current_video_idx: Optional[int] = None
    current_video_path: Optional[Path] = None
    current_frames: Optional[np.ndarray] = None
    current_video_decode_time = 0.0

    for vid_idx, meta, qa in pbar:
        uid = normalize_qa_uid(qa["uid"])
        if args.resume and uid in existing_by_uid and not existing_by_uid[uid].get("error"):
            continue

        t0 = time.time()
        raw_output = ""
        pred_choice = args.fallback_choice
        parse_ok = False
        parse_rule = "not_run"
        error_msg = None
        traceback_str = None
        video_path_str = None
        decode_time = 0.0
        generation_time = 0.0

        try:
            if current_video_idx != vid_idx:
                current_video_idx = vid_idx
                current_video_path = resolve_video_path(meta, video_index)
                video_path_str = str(current_video_path)
                td0 = time.time()
                current_frames = load_video_frames(current_video_path, frame_count)
                current_video_decode_time = time.time() - td0
            else:
                video_path_str = str(current_video_path)
            decode_time = current_video_decode_time
            assert current_frames is not None

            prompt_text = build_prompt(qa, meta, args.prompt_style)
            tg0 = time.time()
            raw_output = generate_answer(
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                video_frames=current_frames,
                prompt_text=prompt_text,
                args=args,
                device=device,
            )
            generation_time = time.time() - tg0
            pred_choice, parse_ok, parse_rule = extract_choice(raw_output, fallback=args.fallback_choice)

        except Exception as exc:  # Continue evaluation; every failure is auditable.
            error_msg = repr(exc)
            traceback_str = traceback.format_exc(limit=20)
            LOGGER.error("Failed uid=%s frame_count=%d error=%s", uid, frame_count, error_msg)
            append_jsonl(
                failure_path,
                {
                    "uid": uid,
                    "video_idx": vid_idx,
                    "frame_count": frame_count,
                    "error": error_msg,
                    "traceback": traceback_str,
                    "meta_preview": {k: meta.get(k) for k in list(meta.keys())[:20]},
                    "qa": qa,
                },
            )

        row = {
            "uid": uid,
            "video_idx": vid_idx,
            "video_path": video_path_str,
            "frame_count": frame_count,
            "question_type": get_question_type(meta, qa),
            "question": qa.get("question", qa.get("query", None)),
            "gt_answer": get_gt_answer(qa),
            "pred_choice": pred_choice,
            "parse_ok": parse_ok,
            "parse_rule": parse_rule,
            "raw_output": raw_output,
            "error": error_msg,
            "decode_time_sec": decode_time,
            "generation_time_sec": generation_time,
            "total_time_sec": time.time() - t0,
        }
        append_jsonl(shard_pred_path, row)
        existing_rows.append(row)
        existing_by_uid[uid] = row

        if rank == 0 and len(existing_rows) % args.save_every == 0:
            latest_rows = list(answers_to_latest_rows(existing_rows).values())
            metrics = write_answers_and_metrics(frame_dir, latest_rows)
            pbar.set_postfix(acc=f"{metrics.get('valid_non_na_acc', 0):.3f}", parse_fail=metrics.get("parse_failures", 0))

        # Keep memory stable for long MotionBench runs.
        if torch.cuda.is_available() and args.cuda_empty_cache_every > 0 and len(existing_rows) % args.cuda_empty_cache_every == 0:
            torch.cuda.empty_cache()

    # In single-process mode, write final answer/metrics now.
    if world_size == 1:
        final_rows = list(answers_to_latest_rows(existing_rows).values())
        metrics = write_answers_and_metrics(frame_dir, final_rows)
        LOGGER.info("Frame count=%d final metrics: %s", frame_count, json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    LOGGER.info("Frame count=%d rank=%d shard complete: %s", frame_count, rank, shard_pred_path)
    return None


def merge_frame_count_outputs(frame_dir: Path, world_size: Optional[int] = None) -> Dict[str, Any]:
    paths = sorted(frame_dir.glob("predictions.rank*.jsonl"))
    if not paths and (frame_dir / "predictions.jsonl").exists():
        paths = [frame_dir / "predictions.jsonl"]
    if world_size is not None and world_size > 1 and len(paths) < world_size:
        LOGGER.warning("Only found %d/%d shard prediction files under %s", len(paths), world_size, frame_dir)

    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = str(row["uid"])
                # Prefer non-error rows if duplicate UID exists.
                if uid not in merged or (merged[uid].get("error") and not row.get("error")):
                    merged[uid] = row

    merged_rows = list(merged.values())
    pred_path = frame_dir / "predictions.merged.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in sorted(merged_rows, key=lambda x: str(x.get("uid"))):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = write_answers_and_metrics(frame_dir, merged_rows)
    return metrics


def write_summary(output_dir: Path, all_metrics: Dict[int, Dict[str, Any]], args: argparse.Namespace) -> None:
    if not all_metrics:
        return
    metric_keys = sorted({k for m in all_metrics.values() for k in m.keys()})
    preferred = [
        "acc",
        "answered_acc",
        "valid_non_na_acc",
        "right_num",
        "total_qa_num",
        "valid_non_na_right",
        "valid_non_na_total",
        "parse_failures",
        "errors",
    ]
    ordered_keys = preferred + [k for k in metric_keys if k not in preferred]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_count"] + ordered_keys)
        writer.writeheader()
        for frame_count in sorted(all_metrics):
            row = {"frame_count": frame_count}
            row.update(all_metrics[frame_count])
            writer.writerow(row)
    atomic_write_json(output_dir / "summary.json", {str(k): v for k, v in sorted(all_metrics.items())}, indent=2)


# -----------------------------
# CLI
# -----------------------------


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FlashVID/LLaVA-OneVision on MotionBench.")

    # Dataset and output.
    parser.add_argument("--meta-file", type=Path, required=True, help="MotionBench video_info.meta.jsonl")
    parser.add_argument("--video-roots", type=Path, nargs="+", required=True, help="One or more roots containing MotionBench videos")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/motionbench_flashvid"))
    parser.add_argument("--frame-counts", type=int, nargs="+", default=[8, 20, 33, 40])
    parser.add_argument("--limit", type=int, default=None, help="Debug: evaluate only first N QA items before sharding")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata/video resolution and prompts without loading the model")
    parser.add_argument("--disable-recursive-index", action="store_true", help="Skip recursive video indexing; only direct paths will work")

    # Model.
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device-map", type=str, default="auto", help="rank_device, auto, cuda:0, etc.")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--conv-template", type=str, default="qwen_1_5")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=1)

    # FlashVID.
    parser.add_argument("--enable-flashvid", action="store_true", help="Wrap LLaVA-OneVision with FlashVID")
    parser.add_argument("--retention-ratio", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--token-selection-method", type=str, default="attn_div_v2", choices=["attn", "div", "attn_div", "attn_div_v2"])
    parser.add_argument("--temporal-threshold", type=float, default=0.80)
    parser.add_argument("--do-segment", type=str2bool, default=True)
    parser.add_argument("--segment-threshold", type=float, default=0.90)
    parser.add_argument("--min-segment-num", type=int, default=8)
    parser.add_argument("--complementary-segment", type=str2bool, default=True)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument(
        "--inner-llm-pruning",
        action="store_true",
        help="Enable FlashVID/FastV-style pruning inside the LLM. Default off for clean vision-side FlashVID evaluation.",
    )
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.30)

    # Prompt/extraction.
    parser.add_argument("--prompt-style", type=str, default="strict", choices=["strict", "motionbench", "plain"])
    parser.add_argument("--fallback-choice", type=str, default="A", choices=list(CHOICES))

    # Runtime/resume.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip UIDs already present in prediction logs")
    parser.add_argument("--overwrite", action="store_true", help="Delete previous shard prediction/failure logs for this run")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--cuda-empty-cache-every", type=int, default=50)
    parser.add_argument("--merge-only", action="store_true", help="Only merge shard files and compute metrics")

    # --- sampled-variant flags (not in the original) ---
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    # NOTE: eval_flashvid.py already defines --seed, so we must not re-add it
    # (argparse raises "conflicting option string: --seed").
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _SAMP.update(temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)
    import random as _rnd
    import numpy as _np
    _rnd.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rank, world_size, local_rank = get_dist_info()

    if torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()

        # In the recommended Slurm launch, each task sees exactly one GPU.
        # Then every process should use cuda:0 inside its own CUDA_VISIBLE_DEVICES namespace.
        if visible_gpu_count == 1:
            device = "cuda:0"
        else:
            device = f"cuda:{local_rank % visible_gpu_count}"

        torch.cuda.set_device(torch.device(device))
    else:
        device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(args.output_dir / "run.log", rank=rank)
    set_reproducible(args.seed + rank)

    run_meta = RunMetadata(
        command=" ".join(sys.argv),
        model_path=args.model_path,
        enable_flashvid=args.enable_flashvid,
        frame_counts=args.frame_counts,
        prompt_style=args.prompt_style,
        retention_ratio=args.retention_ratio,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        do_segment=args.do_segment,
        segment_threshold=args.segment_threshold,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        inner_llm_pruning=args.inner_llm_pruning,
        pruning_layer=args.pruning_layer,
        llm_retention_ratio=args.llm_retention_ratio,
        token_selection_method=args.token_selection_method,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    if is_rank0(rank):
        atomic_write_json(args.output_dir / "run_metadata.json", asdict(run_meta), indent=2)

    LOGGER.info("rank=%d world_size=%d local_rank=%d device=%s", rank, world_size, local_rank, device)
    LOGGER.info("Reading metadata: %s", args.meta_file)
    meta_rows = read_jsonl(args.meta_file)
    LOGGER.info("Loaded %d video metadata rows and %d QA items", len(meta_rows), sum(1 for _ in iter_qas(meta_rows)))

    if args.merge_only:
        all_metrics: Dict[int, Dict[str, Any]] = {}
        for fc in args.frame_counts:
            metrics = merge_frame_count_outputs(args.output_dir / f"frames_{fc:02d}", world_size=world_size)
            all_metrics[fc] = metrics
            LOGGER.info("Merged frame_count=%d metrics=%s", fc, json.dumps(metrics, ensure_ascii=False))
        if is_rank0(rank):
            write_summary(args.output_dir, all_metrics, args)
        return

    video_index = build_video_index(args.video_roots, disable_recursive_index=args.disable_recursive_index)

    tokenizer = model = image_processor = max_length = None
    if not args.dry_run:
        tokenizer, model, image_processor, max_length = load_llava_onevision(args, device=device)
        device_obj = get_model_device(model, fallback=device)
        LOGGER.info("Model input device resolved to %s", device_obj)
    else:
        device_obj = torch.device(device)

    all_metrics: Dict[int, Dict[str, Any]] = {}
    for fc in args.frame_counts:
        metrics = evaluate_one_frame_count(
            frame_count=fc,
            meta_rows=meta_rows,
            video_index=video_index,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            args=args,
            device=device_obj,
            rank=rank,
            world_size=world_size,
        )
        if metrics is not None and is_rank0(rank):
            all_metrics[fc] = metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # In distributed mode, each rank writes shards. Run --merge-only after all ranks finish.
    if world_size == 1 and is_rank0(rank):
        write_summary(args.output_dir, all_metrics, args)
        LOGGER.info("Done. Summary: %s", args.output_dir / "summary.csv")
    elif is_rank0(rank):
        LOGGER.info("Distributed shards complete. Now run the same command with --merge-only to produce final answers/metrics.")


if __name__ == "__main__":
    main()

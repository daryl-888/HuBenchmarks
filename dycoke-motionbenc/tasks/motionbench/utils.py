import re
import os
from loguru import logger as eval_logger

VIDEO_BASE_PATH = "/project/rhu/MotionBench_Data/MotionBench"

MOTION_CATEGORIES = [
    "Action Order",
    "Camera Motion",
    "Location-related Motion",
    "Motion-related Objects",
    "Motion Recognition",
    "Repetition Count",
]


def motionbench_doc_to_visual(doc):
    video_path = doc["video_path"]
    # search both subdirectories
    for subdir in ["self-collected", "public-dataset"]:
        full_path = os.path.join(VIDEO_BASE_PATH, subdir, video_path)
        if os.path.exists(full_path):
            return [full_path]
    # fallback
    return [os.path.join(VIDEO_BASE_PATH, video_path)]


def motionbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["qa"][0]["question"]
    post_prompt = "\nAnswer with the option's letter from tshe given choices directly."
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)
    return f"{question}{post_prompt}"


def motionbench_doc_to_target(doc):
    return doc["qa"][0]["answer"]


def motionbench_process_results(doc, results):
    prediction = results[0] if results else ""
    match = re.search(r'\b([A-D])\b', prediction.upper())
    pred_letter = match.group(1) if match else prediction.strip().upper()[:1]
    ground_truth = doc["qa"][0]["answer"].strip().upper()
    correct = int(pred_letter == ground_truth)
    category = doc.get("question_type", "Unknown")
    return {
        "motionbench_accuracy": correct,
        "category": category,
    }


def motionbench_aggregate_results(results):
    total = len(results)
    if total == 0:
        return 0.0
    correct = sum(results)
    acc = correct / total
    eval_logger.info(f"Accuracy: {correct}/{total} = {acc:.4f}")
    return acc

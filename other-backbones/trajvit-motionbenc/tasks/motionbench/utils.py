import re
import os
from loguru import logger as eval_logger

VIDEO_BASE_PATH = "/project/rhu/MotionBench_Data/MotionBench"

def motionbench_doc_to_visual(doc):
    video_path = doc["video_path"]
    for subdir in ("self-collected", "public-dataset"):
        full_path = os.path.join(VIDEO_BASE_PATH, subdir, video_path)
        if os.path.exists(full_path):
            return [full_path]
    return []


def motionbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["qa"][0]["question"]
    post_prompt = "\nAnswer with the option's letter from the given choices directly."
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

    # Skip NA samples — unanswerable questions that have no valid A-D answer
    if ground_truth == "NA":
        return {
            "motionbench_accuracy": None,
            "category": doc.get("question_type", "Unknown"),
        }

    correct = int(pred_letter == ground_truth)
    category = doc.get("question_type", "Unknown")
    return {
        "motionbench_accuracy": correct,
        "category": category,
    }


def motionbench_aggregate_results(results):
    # Filter out NA samples
    valid_results = [r for r in results if r is not None]
    total = len(valid_results)
    if total == 0:
        return 0.0
    correct = sum(valid_results)
    acc = correct / total
    eval_logger.info(f"Accuracy: {correct}/{total} = {acc:.4f} (skipped {len(results) - total} NA samples)")
    return acc

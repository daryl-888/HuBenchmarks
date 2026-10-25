#!/usr/bin/env python3
"""
Prepare MotionBench metadata for DyTo's OWN scripts (no DyTo source edited).

This is data plumbing only:

  video_info.meta.jsonl (MotionBench)
        |
        v   convert()
  dyto_gt.json  -- schema DyTo's run_inference_multiple_choice_qa.py reads:
                       task_name, video_name, question_id, question,
                       answer_number, candidates, answer

DyTo's MCQA entrypoint is invoked UNMODIFIED by eval_dyto_official.py; this
module only reshapes the dataset so that script can consume it. It never
touches a file inside the DyTo tree.
"""
import argparse
import json
import os


def find_video(video_base: str, rel: str):
    """Resolve a MotionBench video_path under self-collected/public-dataset."""
    for sub in ("self-collected", "public-dataset"):
        p = os.path.join(video_base, sub, rel)
        if os.path.exists(p):
            return p
    return None


def convert(meta_path: str, video_base: str, gt_out: str, limit=None):
    """Read MotionBench meta jsonl and write DyTo-format GT JSON.

    MotionBench per-row fields:
        video_path, question_type, qa[0].{question, answer, options:{A..D}}
    DyTo run_inference_multiple_choice_qa.py per-sample fields:
        task_name, video_name, question_id, question,
        answer_number, candidates, answer
    """
    samples = []
    resolved = 0
    with open(meta_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and i >= limit:
                break
            m = json.loads(line)
            qa = (m.get("qa") or [{}])[0]
            letter_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            ans_letter = (qa.get("answer") or "NA").strip().upper()
            options = qa.get("options") or {}
            candidates = [options.get(ch, "") for ch in "ABCD"]
            # Keep only the options that are actually present, in A..D order.
            candidates = [c for c in candidates if c]
            if not candidates:
                # No option text (rare / NA rows) - placeholder so DyTo's
                # option-formatter does not crash before the defect we verify.
                candidates = ["A", "B", "C", "D"]

            video_name = m.get("video_path", "")
            full = find_video(video_base, video_name)
            if full:
                video_name = full
                resolved += 1

            samples.append({
                "task_name": m.get("question_type", "Unknown"),
                "video_name": video_name,
                "question_id": i,
                "question": qa.get("question", ""),
                "answer_number": letter_index.get(ans_letter, 0),
                "candidates": candidates,
                "answer": options.get(ans_letter, "") or ans_letter,
            })

    out_dir = os.path.dirname(os.path.abspath(gt_out))
    os.makedirs(out_dir, exist_ok=True)
    with open(gt_out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[prepare] wrote {len(samples)} samples -> {gt_out} "
          f"({resolved} videos resolved on disk)", flush=True)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="MotionBench video_info.meta.jsonl")
    ap.add_argument("--video-base", required=True,
                    help="$MOTIONBENCH root (contains self-collected/, public-dataset/)")
    ap.add_argument("--gt-out", required=True,
                    help="DyTo GT JSON output path")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only convert the first N rows (smoke)")
    args = ap.parse_args()
    convert(args.meta, args.video_base, args.gt_out, args.limit)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DyTo — Unmodified Upstream Verification wrapper (wrapper ONLY; no DyTo edits).

This script does NOT import `dyto.*`, does NOT call model.generate(...), and
does NOT edit a single file in the DyTo tree. It is plumbing that:

  1. converts MotionBench metadata -> DyTo GT JSON     (prepare_dyto_gt.py)
  2. writes DyTo's own exp_config YAML
  3. launches DyTo's OWN orchestrator:

        python run_inference.py --exp_config <cfg>

     which in turn dispatches (DyTo's own bash string) to DyTo's OWN

        python run_inference_multiple_choice_qa.py ...

  4. captures stdout/stderr/exit code
  5. if predictions were produced at all (expected: none), scores them with
     the repo's standard letter / NA protocol
  6. writes verification.json recording WHICH documented released-code defect
     reproduced (or that upstream behaviour changed)

Expectation (see README.md): the run terminates at the first released-code
defect -- `tw_finch` absent from finch-clust==0.2.0 (UPSTREAM_DEFECTS.md 1.1)
or the missing `return` in finch_cluster() (1.2). That failure IS the
verification.
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_dyto_gt import convert  # noqa: E402


def write_exp_config(cfg_path, dyto_root, gt_json, out_dir, model_path,
                     num_frames, temporal_aggregation, rope_scaling):
    """Write the YAML/JSON DyTo's own run_inference.py consumes.

    run_inference.py pops SCRIPT, turns every other key into a `k=v` env-var
    string, and `bash -c`s SCRIPT with those vars. SCRIPT below is therefore a
    full command referencing those env vars -- all DyTo machinery, untouched.
    """
    out_name = "dyto_official"
    script = (
        "python run_inference_multiple_choice_qa.py "
        '--video_dir "$VIDEO_DIR" '
        '--gt_file "$GT_FILE" '
        '--output_dir "$OUTPUT_DIR" '
        '--output_name "$OUTPUT_NAME" '
        '--model_path "$MODEL_PATH" '
        '--conv_mode "$CONV_MODE" '
        '--num_frames "$NUM_FRAMES" '
        '--temporal_aggregation "$TEMPORAL_AGGREGATION" '
        '--rope_scaling_factor "$ROPE_SCALING_FACTOR"'
    )
    cfg = {
        "CONFIG_NAME": "auto",
        "SCRIPT": [script],
        "VIDEO_DIR": os.environ.get("MOTIONBENCH",
                                    "/project/rhu/MotionBench_Data/MotionBench"),
        "GT_FILE": gt_json,
        "OUTPUT_DIR": out_dir,
        "OUTPUT_NAME": out_name,
        "MODEL_PATH": model_path,
        "CONV_MODE": "vicuna_v1",
        "NUM_FRAMES": num_frames,
        "TEMPORAL_AGGREGATION": temporal_aggregation,
        "ROPE_SCALING_FACTOR": rope_scaling,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        # JSON is valid YAML; avoids a pyyaml dependency in the wrapper.
        json.dump(cfg, f, indent=2)
    return out_name, cfg


def run_dyto(dyto_root, cfg_path, python_bin, log_path):
    """Invoke DyTo's own run_inference.py; return (returncode, log_text)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{dyto_root}:{env.get('PYTHONPATH', '')}"

    cmd = [python_bin, os.path.join(dyto_root, "run_inference.py"),
           "--exp_config", cfg_path]
    print(f"[wrapper] launching DyTo's own entrypoint:\n  {' '.join(cmd)}",
          flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=dyto_root, env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    return proc.returncode, log_text


def score_output(out_json, meta_path, out_dir, limit):
    """Score DyTo output if it exists (expected: it does not)."""
    results = []
    scores = []
    with open(out_json, encoding="utf-8") as f:
        dyto_rows = [json.loads(l) for l in f if l.strip()]
    meta = []
    with open(meta_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and i >= limit:
                break
            meta.append(json.loads(line))

    by_id = {str(r.get("id")): r for r in dyto_rows}
    for idx, m in enumerate(meta):
        pred = by_id.get(str(idx), {}).get("pred", "")
        truth = (m.get("qa") or [{}])[0].get("answer", "NA").strip().upper()
        sc = None
        if truth != "NA":
            mm = re.search(r"\b([A-D])\b", pred.upper())
            pl = mm.group(1) if mm else pred.strip().upper()[:1]
            sc = int(pl == truth)
        results.append({
            "idx": idx,
            "video_path": m.get("video_path", ""),
            "question_type": m.get("question_type", "Unknown"),
            "ground_truth": truth,
            "prediction": pred,
            "correct": sc,
        })
        if sc is not None:
            scores.append(sc)

    out_file = os.path.join(out_dir, "results.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    total = len(scores)
    correct = sum(scores)
    acc = correct / total if total else 0.0
    summary = {
        "accuracy": acc, "correct": correct,
        "total_scoreable": total,
        "total_na_skipped": len(results) - total,
        "total_samples": len(results),
        "note": "UNEXPECTED: unmodified DyTo produced predictions. "
                "Re-open docs/UPSTREAM_DEFECTS.md before trusting.",
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return acc, len(results)


def classify(log_text):
    """Map observed traceback to the documented released-code defect."""
    if "unexpected keyword argument 'tw_finch'" in log_text:
        return ("DEFECT-1.1-REPRODUCED",
                "FINCH() from finch-clust==0.2.0 rejects tw_finch "
                "(UPSTREAM_DEFECTS.md 1.1). Released code cannot run.")
    if "NoneType" in log_text and "shape" in log_text:
        return ("DEFECT-1.2-REPRODUCED",
                "finch_cluster() returned None; missing return statement "
                "(UPSTREAM_DEFECTS.md 1.2). Released code cannot run.")
    if "cannot import name" in log_text or "No module named" in log_text:
        return ("DEFECT-2.1-REPRODUCED",
                "Import failure in the released package "
                "(UPSTREAM_DEFECTS.md 2 row 1). Released code cannot run.")
    if "Error" in log_text or "Traceback" in log_text:
        return ("DEFECT-OTHER",
                "Run terminated with an unclassified traceback. "
                "See --workdir/dyto_run.log.")
    return ("PRODUCED-PREDICTIONS",
            "No traceback observed; predictions may exist. Behaviour of the "
            "unmodified upstream tree appears to have changed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="MotionBench video_info.meta.jsonl")
    ap.add_argument("--video-base", required=True, help="$MOTIONBENCH root")
    ap.add_argument("--dyto-root", required=True,
                    help="Pristine DyTo checkout (see verify_upstream.sh)")
    ap.add_argument("--python", default=os.environ.get(
        "DYTO_PYTHON", "/project/rhu/dpalfaro/conda/envs/dyto_v/bin/python3"))
    ap.add_argument("--model-path", required=True,
                    help="LLaVA-NeXT Vicuna-7B weights")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--temporal-aggregation",
                    default="spatial_tome_finch_dynamic_all_frms")
    ap.add_argument("--rope-scaling", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    gt_json = os.path.join(args.workdir, "dyto_gt.json")
    cfg_path = os.path.join(args.workdir, "exp_config.json")
    log_path = os.path.join(args.workdir, "dyto_run.log")

    print("[wrapper] converting MotionBench meta -> DyTo GT JSON ...", flush=True)
    convert(args.meta, args.video_base, gt_json, limit=args.limit)

    out_name, _cfg = write_exp_config(
        cfg_path, args.dyto_root, gt_json, args.workdir, args.model_path,
        args.num_frames, args.temporal_aggregation, args.rope_scaling)
    print(f"[wrapper] wrote {cfg_path}", flush=True)

    rc, log_text = run_dyto(args.dyto_root, cfg_path, args.python, log_path)
    tail = log_text[-4000:]

    out_json = os.path.join(args.workdir, f"{out_name}.json")
    status, message = classify(log_text)

    verification = {
        "verification": "dyto-unmodified-upstream",
        "expected": "FAIL at released-code defect "
                    "(tw_finch 1.1 and/or finch_cluster return 1.2)",
        "observed_status": status,
        "message": message,
        "exit_code": rc,
        "dyto_pin": "see upstream_tree.sha1.txt (verify_upstream.sh)",
        "log_tail": tail,
    }

    if os.path.exists(out_json):
        acc, n = score_output(out_json, args.meta, args.workdir, args.limit)
        verification["scores_written"] = True
        verification["scored"] = {"total": n, "accuracy": acc}
    else:
        verification["scores_written"] = False

    with open(os.path.join(args.workdir, "verification.json"), "w",
              encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    print(f"\n=== VERIFICATION RESULT ===", flush=True)
    print(f"  status : {status}", flush=True)
    print(f"  exit   : {rc}", flush=True)
    print(f"  log    : {log_path}", flush=True)

    if status == "PRODUCED-PREDICTIONS":
        sys.exit(2)  # upstream behaviour changed; fail loudly
    sys.exit(0)


if __name__ == "__main__":
    main()
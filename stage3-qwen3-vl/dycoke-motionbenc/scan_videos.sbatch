#!/bin/bash
# Scans all unique MotionBench videos for NFS stale-handle issues.
# Prints BAD_HANG (read blocks) or MISSING (not found in either subdir).
# Run this BEFORE inference jobs. Move any BAD_HANG files to
# /project/rhu/dpalfaro/bad_videos/ so utils.py skips them automatically.
#SBATCH -J s3_dycoke_run_scan_videos
#SBATCH -o /project/rhu/dpalfaro/results/s3_dycoke_run_scan_videos_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/s3_dycoke_run_scan_videos_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --gpus=ada:1
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

module purge
module load Miniforge3/py3.10

/project/rhu/dpalfaro/conda/envs/dycoke11/bin/python3 - << 'EOF'
import json
import os
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

VIDEO_BASE = "/project/rhu/MotionBench_Data/MotionBench"
META      = f"{VIDEO_BASE}/video_info.meta.jsonl"
TIMEOUT   = 10  # seconds per probe — full read; NFS stale blocks hang indefinitely
WORKERS   = 8   # lower parallelism since each probe reads the whole file

def probe(video_path):
    """Read the entire video file. Returns (path, status).
    Status: 'OK', 'BAD_HANG', or 'MISSING'.
    Reading 1 byte is not enough — NFS stale handles can affect blocks beyond
    the first sector while stat() and small reads still succeed from cache.
    """
    for subdir in ("self-collected", "public-dataset"):
        full = os.path.join(VIDEO_BASE, subdir, video_path)
        proc = subprocess.Popen(
            ["dd", f"if={full}", "bs=4M", "of=/dev/null"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=TIMEOUT)
            if proc.returncode == 0:
                return video_path, "OK"
            # returncode != 0 means fast fail (file not found in this subdir)
        except subprocess.TimeoutExpired:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return video_path, "BAD_HANG"
    return video_path, "MISSING"

# Collect unique video paths from the JSONL
print(f"Reading {META} ...")
with open(META) as f:
    unique_paths = list({json.loads(line)["video_path"] for line in f})
print(f"Found {len(unique_paths)} unique videos. Scanning with {WORKERS} workers ...")

ok = bad_hang = missing = 0
bad_list = []

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(probe, p): p for p in unique_paths}
    done = 0
    for fut in as_completed(futures):
        path, status = fut.result()
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(unique_paths)} scanned ...", flush=True)
        if status == "OK":
            ok += 1
        elif status == "BAD_HANG":
            bad_hang += 1
            bad_list.append(path)
            print(f"BAD_HANG: {path}", flush=True)
        else:
            missing += 1
            print(f"MISSING:  {path}", flush=True)

print()
print(f"=== Scan complete ===")
print(f"  OK:        {ok}")
print(f"  BAD_HANG:  {bad_hang}")
print(f"  MISSING:   {missing}")
print()
if bad_list:
    print("Move these to /project/rhu/dpalfaro/bad_videos/ before running inference:")
    for p in sorted(bad_list):
        print(f'    mv .../self-collected/{p!r} /project/rhu/dpalfaro/bad_videos/')
else:
    print("No BAD_HANG files found — safe to run inference.")
EOF

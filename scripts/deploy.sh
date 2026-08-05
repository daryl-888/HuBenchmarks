#!/usr/bin/env bash
#
# deploy.sh — sync local eval wrappers to Carya, killing the "stale sbatch" trap.
#
# The problem this solves: `git push` does NOT update Carya, and there is no
# single mirror — the cluster has both the new stage{1,2,3}/other-backbones
# layout AND leftover flat *-motionbenc/ dirs. Editing a file locally and
# rerunning gives identical failures because the OLD copy is what the job runs.
#
# This script rsyncs the four wrapper trees from this repo to their exact
# counterparts on Carya. It NEVER touches the method source forks (DyCoke/,
# HoliTom/, MDP3/, LLaVA/, ...) or weights/ — those are managed by hand.
#
# Usage:
#   scripts/deploy.sh            # dry-run: show what WOULD change, touch nothing
#   scripts/deploy.sh --push     # actually sync
#   scripts/deploy.sh --push stage1-llava-ov          # sync one tree only
#   scripts/deploy.sh --push stage1-llava-ov/dycoke-motionbenc   # one wrapper
#
# Exit status is rsync's, so it composes in Makefiles / CI.

set -euo pipefail

CARYA_USER="dpalfaro"
CARYA_HOST="carya.rcdc.uh.edu"
CARYA_CODE="/project/rhu/dpalfaro/code"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The only trees we ever sync. Local path == remote path (same basename under code/).
TREES=(
  "stage1-llava-ov"
  "stage2-llava-video"
  "stage3-qwen3-vl"
  "other-backbones"
)

# Never ship these — junk, caches, or results that belong on the cluster only.
EXCLUDES=(
  --exclude '.git'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  --exclude '*.out'
  --exclude '*.err'
  --exclude 'results/'
  --exclude 'logs/'
)

MODE="dry"          # dry | push
TARGETS=()

for arg in "$@"; do
  case "$arg" in
    --push)  MODE="push" ;;
    --dry)   MODE="dry" ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)       TARGETS+=("$arg") ;;
  esac
done

# If no explicit target dirs given, sync all four trees.
if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=("${TREES[@]}")
fi

# Prefer rsync (fast, itemized, --delete). Fall back to a tar|ssh pipe when the
# LOCAL machine has no rsync (Linux Mint often doesn't). The tar path can't do a
# true dry-run or --delete, so it just lists what it would ship instead.
if command -v rsync >/dev/null 2>&1; then
  XFER="rsync"
else
  XFER="tar"
  echo ">>> NOTE: local rsync not found — using tar|ssh fallback."
  echo ">>>       (no --delete: files removed locally won't be removed on Carya;"
  echo ">>>        install rsync locally for full mirroring: sudo apt install rsync)"
  echo
fi

RSYNC_FLAGS=(-az --itemize-changes --delete-after)
if [ "$MODE" = "dry" ]; then
  RSYNC_FLAGS+=(--dry-run)
  echo ">>> DRY RUN — no files changed. Re-run with --push to apply."
  echo
fi

# tar-pipe deploy of one dir: streams local tree into remote dir over ssh.
tar_push() {
  local src="$1" dest="$2"
  if [ "$MODE" = "dry" ]; then
    echo "would ship (tar) these files:"
    ( cd "$src" && tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='*.out' --exclude='*.err' --exclude='results' --exclude='logs' \
        -cvf /dev/null . ) 2>&1 | sed 's/^/    /'
    return 0
  fi
  ( cd "$src" && tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='*.out' --exclude='*.err' --exclude='results' --exclude='logs' \
      -czf - . ) \
    | ssh -o BatchMode=yes "$CARYA_USER@$CARYA_HOST" "mkdir -p '$dest' && tar -xzf - -C '$dest'"
}

fail=0
for t in "${TARGETS[@]}"; do
  # Strip any trailing slash for consistent path math.
  t="${t%/}"
  local_path="$REPO_ROOT/$t"
  if [ ! -d "$local_path" ]; then
    echo "!!! skip: '$t' is not a local directory under $REPO_ROOT" >&2
    fail=1
    continue
  fi

  # Guard: only allow paths that start with one of the four sanctioned trees.
  ok=0
  for tree in "${TREES[@]}"; do
    case "$t" in "$tree"|"$tree"/*) ok=1 ;; esac
  done
  if [ "$ok" -eq 0 ]; then
    echo "!!! refuse: '$t' is outside the sanctioned wrapper trees (${TREES[*]})" >&2
    fail=1
    continue
  fi

  remote_path="$CARYA_CODE/$t"
  echo "=== $t  ->  $CARYA_HOST:$remote_path"
  # Ensure the remote parent exists (first-time sync of a nested wrapper).
  if [ "$MODE" = "push" ]; then
    ssh -o BatchMode=yes "$CARYA_USER@$CARYA_HOST" "mkdir -p '$(dirname "$remote_path")'"
  fi
  if [ "$XFER" = "rsync" ]; then
    # rsync semantics: trailing slash on source = "contents of", so mirror dir->dir.
    rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" \
      "$local_path/" \
      "$CARYA_USER@$CARYA_HOST:$remote_path/" || fail=1
  else
    tar_push "$local_path" "$remote_path" || fail=1
  fi
  echo
done

if [ "$MODE" = "dry" ]; then
  echo ">>> DRY RUN complete. Nothing was changed on Carya."
fi
exit "$fail"

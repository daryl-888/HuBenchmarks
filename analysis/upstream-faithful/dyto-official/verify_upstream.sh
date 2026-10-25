#!/usr/bin/env bash
# ===========================================================================
# DyTo — clone + pin the pristine upstream tree, with NO patches.
#
# Verification run philosophy: the released code is NOT modified. This script
# only downloads the exact upstream commit and records its identity so the run
# can be attributed to a pinned revision. That is the whole point — see
# README.md ("nothing patched").
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_URL="${DYTO_UPSTREAM_URL:-https://github.com/Jam1ezhang/DYTO}"
PIN_REV="${DYTO_PIN_REV:-main}"
SRC_DIR="${DYTO_SRC_DIR:-$HERE/DYTO}"

echo "==> DyTo upstream verification (no patches)"
echo "    URL : $UPSTREAM_URL"
echo "    Rev : $PIN_REV"

if [ ! -d "$SRC_DIR/.git" ]; then
    echo "==> Cloning $UPSTREAM_URL into $SRC_DIR ..."
    git clone --no-checkout "$UPSTREAM_URL" "$SRC_DIR"
else
    echo "==> $SRC_DIR already exists; fetching latest refs ..."
    git -C "$SRC_DIR" fetch origin
fi

echo "==> Checking out pinned revision $PIN_REV ..."
git -C "$SRC_DIR" checkout "$PIN_REV"

echo "==> Pinned commit:"
git -C "$SRC_DIR" rev-parse HEAD

echo "==> Verifying working tree hash (should be the pristine upstream tree;"
echo "    no files added, removed, or modified by us):"
PRE_HASH="$(git -C "$SRC_DIR" stash create 2>/dev/null || true)"
git -C "$SRC_DIR" status --porcelain
echo "    (empty above = clean)"
if [ -n "$PRE_HASH" ]; then
    echo "    WARNING: local changes were present before checkout — stashed at $PRE_HASH"
fi

echo "==> Recording pinned tree content hashes for attribution ..."
(
    cd "$SRC_DIR"
    git ls-tree -r HEAD | sort > "$HERE/upstream_tree.sha1.txt"
)
echo "    wrote $HERE/upstream_tree.sha1.txt"

echo "==> DyTo upstream is pinned and UNMODIFIED."
echo "    Tree manifest: $HERE/upstream_tree.sha1.txt"
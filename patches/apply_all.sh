#!/usr/bin/env bash
#
# apply_all.sh — reconstruct the exact method source trees we ran.
#
# Each method wraps its authors' original repository. We pin the upstream commit
# (config/paths.sh: $SHA_*) and apply our modifications as real unified diffs
# (patches/diffs/*.patch). Together those two things reproduce the source that
# produced the published results, even if upstream later rewrites history.
#
# Usage:
#   source config/paths.sh            # or your edited paths.local.sh
#   patches/apply_all.sh --check      # dry-run: verify SHAs + patch applicability
#   patches/apply_all.sh              # actually check out SHAs and apply patches
#
# Only DyCoke, HoliTom and PruneVid carry patches; MDP3, VisionZip and FastV are
# used unmodified at their pinned commits. DyTo has no upstream git history in
# our copy and is vendored as plain files (see config/paths.sh).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFS="$HERE/diffs"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# repo-dir-var : pinned-sha-var : patch file (empty = unmodified upstream)
ENTRIES=(
  "SRC_DYCOKE:SHA_DYCOKE:DyCoke.patch"
  "SRC_HOLITOM:SHA_HOLITOM:HoliTom.patch"
  "SRC_PRUNEVID:SHA_PRUNEVID:PruneVid.patch"
  "SRC_MDP3:SHA_MDP3:"
  "SRC_VISIONZIP:SHA_VISIONZIP:"
  "SRC_FASTV:SHA_FASTV:"
)

fail=0
for e in "${ENTRIES[@]}"; do
    dirvar="${e%%:*}"; rest="${e#*:}"
    shavar="${rest%%:*}"; patch="${rest#*:}"
    dir="${!dirvar:-}"; sha="${!shavar:-}"
    name="${dirvar#SRC_}"

    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "  SKIP  $name — \$$dirvar not set or missing (source config/paths.sh)"
        continue
    fi
    if [ ! -d "$dir/.git" ]; then
        echo "  WARN  $name — not a git checkout, cannot verify pin $sha"
    else
        cur="$(git -C "$dir" rev-parse HEAD 2>/dev/null | cut -c1-12)"
        if [ "$cur" = "$sha" ]; then
            echo "  OK    $name at pinned $sha"
        elif [ "$CHECK_ONLY" = 1 ]; then
            echo "  DIFF  $name is at $cur, pinned is $sha (would check out)"
        else
            echo "  ...   $name: checking out $sha (was $cur)"
            git -C "$dir" checkout -q "$sha" || { echo "  FAIL  $name checkout"; fail=1; }
        fi
    fi

    [ -z "$patch" ] && continue
    p="$DIFFS/$patch"
    [ -f "$p" ] || { echo "  FAIL  missing $p"; fail=1; continue; }
    if git -C "$dir" apply --check "$p" 2>/dev/null; then
        if [ "$CHECK_ONLY" = 1 ]; then
            echo "        patch $patch applies cleanly"
        else
            git -C "$dir" apply "$p" && echo "        applied $patch" \
                || { echo "  FAIL  applying $patch"; fail=1; }
        fi
    elif git -C "$dir" apply --reverse --check "$p" 2>/dev/null; then
        echo "        patch $patch already applied"
    else
        echo "  FAIL  $patch does not apply to $name at this revision"
        fail=1
    fi
done

echo
[ "$fail" = 0 ] && echo "all good" || echo "some steps failed (see above)"
exit "$fail"

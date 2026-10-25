#!/usr/bin/env bash
#
# fetch_results.sh — pull run results from Carya into a local cache.
#
# Only summary.json + results.jsonl are fetched (a few MB), never the videos or
# weights. Runs that are still queued or in flight are simply skipped, so this is
# safe to run repeatedly while jobs land.
#
# Usage:
#   scripts/fetch_results.sh              # fetch every known run
#   scripts/fetch_results.sh --status     # just show which jobs are done/pending
#   scripts/fetch_results.sh --sweep      # only the retention-sweep runs
#
# Then render (no cluster access needed):
#   python3 scripts/build_retention_tables.py --local > docs/RETENTION_TABLES.md

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$HERE/results-cache"
REMOTE="${CARYA_USER:-dpalfaro}@${CARYA_HOST:-carya.rcdc.uh.edu}"
RES="${CARYA_RESULTS:-/project/rhu/dpalfaro/results}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"

# Every run the tables reference. Sweep runs first, then the fixed-r=0.15 wave
# runs and the two baselines the deltas are measured against.
SWEEP_RUNS="
s1_fastv_r10_run s1_fastv_r25_run s1_flashvid_r10_run s1_flashvid_r25_run
s1_holitom_r10_run s1_holitom_r25_run s1_prunevid_r10_run s1_prunevid_r25_run
s3_fastv_r10_run s3_fastv_r25_run s3_flashvid_r10_run s3_flashvid_r25_run
s3_holitom_r10_run s3_holitom_r25_run s3_prunevid_r10_run s3_prunevid_r25_run
"
BASE_RUNS="
fastv_run1 qwen3vl_baseline_run1
w2_fastv_run w2_flashvid_run w2_holitom_run w2_prunevid_ov_run
w3_fastv_run w3_flashvid_run w3_holitom_run w3_prunevid_run
w3_dycoke_run w3_aim_run w3_mdp3_run w3_videoitg_run w3_visionzip_run w3_sttm_run
ob_dyto_run ob_vicuna_baseline
"

case "${1:-}" in
  --status)
    echo ">>> job states on Carya"
    $SSH "$REMOTE" "squeue -u \$USER -o '%.10i %.22j %.9T %.11M %R' -h 2>/dev/null || true; \
      echo '--- recently completed ---'; \
      sacct -X -n --starttime now-3days --format=JobID%10,JobName%22,State%11,Elapsed 2>/dev/null \
        | grep -E 'sweep|_r10|_r25|vicuna' | head -25"
    exit 0 ;;
  --sweep) RUNS="$SWEEP_RUNS" ;;
  *)       RUNS="$SWEEP_RUNS $BASE_RUNS" ;;
esac

mkdir -p "$CACHE"
have=0; miss=0

# One SSH round-trip to find which runs are actually complete, rather than one
# per run (17+ connections to a busy login node is slow and rude).
# The login node prints a legal banner on every connection, which lands in the
# captured output. Delimit the real payload so banner lines cannot be mistaken
# for run names (this silently reported "0 fetched" until it was fixed).
# $RUNS is defined across multiple lines for readability; collapse it before
# interpolating into the remote command, or the embedded newlines break the loop.
RUNS_ONELINE=$(echo $RUNS)
ready=$($SSH "$REMOTE" "echo __BEGIN__; for d in $RUNS_ONELINE; do \
    [ -s '$RES'/\$d/summary.json ] && echo \$d; done; echo __END__" 2>/dev/null \
    | sed -n '/__BEGIN__/,/__END__/p' | grep -vE '__BEGIN__|__END__')

for d in $RUNS; do
  if echo "$ready" | grep -qx "$d"; then
    mkdir -p "$CACHE/$d"
    if scp -q -o BatchMode=yes -o ConnectTimeout=25 \
        "$REMOTE:$RES/$d/summary.json" "$REMOTE:$RES/$d/results.jsonl" \
        "$CACHE/$d/" 2>/dev/null; then
      printf "  \033[32m✓\033[0m %s\n" "$d"; have=$((have+1))
    else
      printf "  \033[31m!\033[0m %s (copy failed)\n" "$d"; miss=$((miss+1))
    fi
  else
    printf "  \033[90m·\033[0m %s (not finished)\n" "$d"; miss=$((miss+1))
  fi
done

echo
echo "fetched $have run(s) into results-cache/ ; $miss not ready"
echo "render with:  python3 scripts/build_retention_tables.py --local"

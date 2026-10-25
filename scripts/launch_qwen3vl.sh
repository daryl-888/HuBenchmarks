#!/usr/bin/env bash
#
# launch_qwen3vl.sh — staged submission of the Qwen3-VL method ports.
#
# Two groups, because they carry different risk:
#
#   GROUP A (embedding-only): FlashVID, AIM, MDP3, VideoITG.
#     These rank/select on input embeddings or frames, NOT attention weights, so
#     they run under sdpa with no eager-attention issue. Safe to smoke immediately.
#
#   GROUP B (attention-based): FastV, DyCoke, HoliTom.
#     These need layer-K attention. The first smoke ran as a baseline (eager broke
#     generation -> empty output). They are now sdpa + report loudly if attention
#     is unavailable. DO NOT launch these until the FastV diagnostic (currently
#     7776370) confirms `<method>(Qwen3-VL) ACTIVE` with non-empty predictions.
#     If it instead prints "NOT PRUNING", they need a per-module eager swap first.
#
# Usage:
#   scripts/launch_qwen3vl.sh smoke-a      # smoke the 4 embedding-only ports
#   scripts/launch_qwen3vl.sh smoke-b      # smoke the 3 attention ports (only after diag confirms)
#   scripts/launch_qwen3vl.sh full <method>  # full run of one verified method
#
# Every smoke must show "<method>(Qwen3-VL) ACTIVE" AND non-empty predictions AND
# diverge from qwen3vl_baseline_run1 before a full run is submitted.

set -euo pipefail
SSH="ssh -o BatchMode=yes -l dpalfaro carya.rcdc.uh.edu"
CODE=/project/rhu/dpalfaro/code/stage3-qwen3-vl

GROUP_A=(flashvid aim mdp3 videoitg)
GROUP_B=(fastv dycoke holitom)

submit_smoke() {
  local m="$1"
  printf 's3_%-9s ' "$m"
  $SSH "cd /project/rhu/dpalfaro && sbatch $CODE/${m}-motionbenc/smoke_${m}_qwen3vl.sbatch" \
    2>/dev/null | grep -oE '[0-9]+$' || echo FAIL
}

case "${1:-}" in
  smoke-a) for m in "${GROUP_A[@]}"; do submit_smoke "$m"; done ;;
  smoke-b) for m in "${GROUP_B[@]}"; do submit_smoke "$m"; done ;;
  full)
    m="${2:?need a method name}"
    # full run = smoke sbatch minus --limit; build on the fly from the verified smoke
    echo "Submit the full run only after the smoke for $m passed the divergence gate."
    echo "Use the per-method w3_run_${m}.sbatch (generated once its smoke is green)."
    ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac

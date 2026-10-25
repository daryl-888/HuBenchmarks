# --- sanity gate: paste at the END of any eval sbatch, after the python eval call ---
# Turns a silent-wrong run into a loud, recorded failure. Requires check_run.py
# synced to Carya (scripts/deploy pushes it, or scp it to a known path).
#
# Assumes $OUTPUT_DIR is the run's results dir (contains summary.json afterward)
# and $METHOD is the method name (e.g. dycoke, holitom). Adjust var names to match.

CHECK=/project/rhu/dpalfaro/code/HuVLLM_scripts/check_run.py
if python "$CHECK" "$OUTPUT_DIR" --expect-method "$METHOD"; then
    echo "GATE: PASS — result recorded."
else
    echo "GATE: FAIL — run is untrustworthy, see checks above." >&2
    # Rename so a bad run can never be mistaken for a good one in the results dir.
    mv "$OUTPUT_DIR" "${OUTPUT_DIR}.FAILED_GATE" 2>/dev/null || true
    exit 1
fi

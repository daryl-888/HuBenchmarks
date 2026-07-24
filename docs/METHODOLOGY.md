# Methodology

## What a run is

For each of the 8,052 MotionBench samples: sample `--num_frames` frames uniformly,
build the prompt (`<image>\n<question>\nAnswer with the option's letter...`), run
greedy generation (`do_sample=False`, `max_new_tokens=16`), and letter-match the
output against the ground-truth option. `NA` samples (4,034) are excluded from
accuracy. Every run writes:

- `results.jsonl` — one row per sample: `idx`, `video_path`, `question_type`,
  `ground_truth`, `prediction`, `correct`.
- `summary.json` — `accuracy`, `correct`, `total_scoreable`, `per_category`, and a
  `<method>_params` block including **`enabled: true`**.

## Standardized configuration

- **32 frames** for every method.
- **15% visual-token retention** where the method exposes a retention knob
  (HoliTom `RETAIN_RATIO=0.15`, FlashVID `retention_ratio=0.15`, FastV
  `fastv_r=0.85` = keep 15%). Methods without a single retention parameter
  (DyCoke `l/p/k`, STTM threshold, MDP3, AIM, VideoITG) keep their **paper
  defaults** — forcing 15% would break paper-fidelity.
- Greedy decoding, letter-match scoring, `NA` skipped.

## The verification gate — why it exists

The dominant failure mode in this project was **a run that completes and lies**: a
method whose hook silently didn't fire, so the "result" is actually the plain
backbone wearing the method's name. Real cases we hit: FastV's `apply_fastv()` was
a no-op stub; a PruneVID→LLaVA-OV port never pruned; VisionZip returned 100% empty
predictions from an output-slicing bug — all reported plausible-looking numbers.

`scripts/check_run.py` turns these into loud failures. It checks:

1. **Completeness** — all 8,052 samples present (no truncation; there is no
   checkpointing, so a died job restarts from scratch).
2. **NA accounting** — ~4,034 skipped, or the wrong metadata was loaded.
3. **Accuracy band** — inside [0.40, 0.95]; below-floor ⇒ dropped weights / empty output.
4. **Non-degenerate predictions** — not empty, not collapsed to one letter.
5. **`enabled`** — the method's own params block confirms it ran.
6. **Divergence** (`--vs-baseline`) — the decisive check: if predictions are
   **identical to the plain backbone (0/8052 differ)**, the method was a silent
   no-op. This is the only check that caught the FastV stub and the inert
   PruneVID-OV port; both passed everything else.

```bash
python scripts/check_run.py <run_dir> --expect-method <m> --vs-baseline <backbone_run>
# --smoke skips completeness/NA/accuracy for --limit test runs
```

**A method's number is only reported once it passes the divergence gate.** In the
results table, ✅ = gated, 🔄 = running/unverified, ❌ = known no-op/crash.

## Reproducing the verification

Each method's page lists its "ACTIVE" log line — e.g.
`FastV ACTIVE: img_len=6273 keep=941 (dropped 5332)`. Its absence in a run's stderr
means the method did not engage, regardless of what accuracy it reported.

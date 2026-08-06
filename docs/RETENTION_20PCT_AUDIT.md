# 20% retention: which methods can honestly be run there

Checked each method's own published sweep before running anything.

| Method | 20% status | Action |
|---|---|---|
| FlashVID | Authors' exact published point — `RETENTION_RATIOS=(0.10 0.15 0.20 0.25)` on all four of their backbone scripts | Ran (`s1_flashvid_r20_run`, job 7987557) |
| HoliTom | Authors' exact published point — `RETAIN_RATIO=0.20` line in `eval_ov-7b_holitom.sh` | Ran (`s1_holitom_r20_run`, job 7987556) |
| FastV | Free parameter, but authors only publish 12.5/25/50/75%; 20% interpolates, and already sits inside our fully-characterized flat-collapse range (10–75%) | Skipped — redundant |
| PruneVID | Authors publish exactly one `cluster_ratio` value, `0.5`, no sweep at all | Skipped — no authors' basis for 0.2 |
| DyCoke | No single retention knob — authors publish one fixed recipe (`l=3, p=0.7, k=0.7`), not a percentage | Skipped — undefined mapping |
| AIM | No ratio knob at all on LLaVA-OV (hardcoded merge); on Qwen3-VL the knob is our own invented port — authors have no native Qwen3-VL code | Skipped — no authors' basis on either backbone |

Both runs are on LLaVA-OV-7B, gated against `fastv_run1` via `check_run.py --expect-method`.

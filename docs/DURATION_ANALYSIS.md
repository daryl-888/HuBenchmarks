# Is the collapse duration-dependent?

**Finding. No.** FastV's 15.9-point loss on LLaVA-OV is the same on clips under
5 seconds as on clips over 5 seconds. The naive split appears to show a
short-clip penalty; that appearance is entirely question-type mix and a floor
effect. Once question type is held fixed the gap is **+0.09 points**.

This closes a plausible alternative to the density-threshold explanation in
[FASTV_COLLAPSE_ANALYSIS.md](FASTV_COLLAPSE_ANALYSIS.md).

MotionBench durations come from `video_info.duration`, cached to
`results-cache/_video_durations.json`. Median 5.76 s, range 0.60–49.96 s, so a
5 s cut splits 1,745 short against 2,273 long scoreable questions.

Reproduce:

```bash
python3 analysis/duration-split/split_by_duration.py --cut 3 --cut 5 --cut 10
python3 analysis/duration-split/collapse_by_duration.py
python3 analysis/duration-split/duration_within_category.py --cut 5
```

---

## 1. The hypothesis

At a fixed 32 frames, a 3-second clip is sampled at 10.7 fps and a 30-second
clip at 1.1 fps. Adjacent frames in a short clip are near-duplicates, so a
token budget spent on a short clip may buy far less unique information. If that
drives the collapse, short clips should be hit harder.

## 2. The naive split says yes — at 3 s

LLaVA-OV-7B, keep 15%, delta against the same-half backbone:

| Cut | Method | Short Δ | Long Δ | Asymmetry | Interaction χ² |
|---|---|---|---|---|---|
| 3 s | FastV | −19.78 | −14.58 | **−5.20** | 16.28 **signif** |
| 3 s | PruneVID-OV | −18.28 | −13.19 | **−5.09** | 19.14 **signif** |
| 5 s | FastV | −16.91 | −15.09 | −1.82 | 2.95 no |
| 5 s | PruneVID-OV | −15.93 | −13.33 | −2.60 | 5.78 **signif** |

No non-collapsed method shows anything: FlashVID +0.58, DyCoke +1.00,
HoliTom +0.58, all with χ² < 1.

Taken alone this reads as a real short-clip penalty specific to the two
collapsed methods. It is not.

## 3. Two confounds

**Category mix.** The halves are not asking the same questions. At the 3 s cut:

| Category | Short % | Long % | Diff |
|---|---|---|---|
| Motion-related Objects | 33.7 | 11.7 | **+22.0** |
| Location-related Motion | 6.5 | 15.9 | −9.4 |
| Repetition Count | 3.7 | 12.0 | −8.3 |
| Camera Motion | 4.9 | 11.1 | −6.2 |

Duration is largely a proxy for question type. Motion-related Objects — the
category the backbone handles best (71.2%) and where FastV retains most — is
three times as common among short clips.

**Floor effect.** The backbone scores 62.54 on clips under 3 s and 49.39 above.
Any method degenerating toward a fixed answering strategy loses more points
wherever the baseline was better, with no duration mechanism involved.

## 4. The measure that is not confounded

D-rate — share of answers on option D — is the collapse signature (~47% against
a 23.6% ground-truth rate) and does not depend on baseline skill. If short clips
collapsed harder, their D-rate would be higher:

| Run | D short (<3 s) | D long (≥3 s) | Diff |
|---|---|---|---|
| *LLaVA-OV baseline* | *19.5* | *19.2* | *+0.3* |
| FastV keep 10% | 44.6 | 47.8 | **−3.2** |
| FastV keep 15% | 44.6 | 47.5 | **−2.9** |
| FastV keep 25% | 45.0 | 47.7 | **−2.8** |
| PruneVID-OV keep 15% | 41.3 | 47.6 | **−6.3** |

The collapse is *milder* on short clips, not worse — the opposite of what the
larger accuracy drop suggested. That drop is the floor effect.

## 5. Question type controlled

Holding category fixed and splitting within it at 5 s, LLaVA-OV keep 15%:

| Category | n (S/L) | FastV Δ short | FastV Δ long | Asym |
|---|---|---|---|---|
| Action Order | 203/316 | −6.90 | −8.23 | +1.33 |
| Camera Motion | 137/248 | −20.44 | −9.27 | −11.16 |
| Location-related Motion | 175/371 | −21.71 | −21.29 | −0.42 |
| Motion Recognition | 697/781 | −19.66 | −22.92 | +3.26 |
| Motion-related Objects | 448/242 | −17.63 | −16.94 | −0.69 |
| Repetition Count | 85/315 | +1.18 | +1.59 | −0.41 |
| **Weighted mean** | | | | **+0.09** |

The asymmetry disappears. Same control on the others: PruneVID-OV **−0.44**,
FlashVID **+0.54**, DyCoke **+1.25** — all inside noise.

Camera Motion is the one cell that still looks like an effect (−11.16, and
PruneVID-OV −8.80 in the same direction). It does not survive testing:
interaction χ² = 2.41 for FastV and 1.44 for PruneVID-OV, against a 3.84
threshold. With six categories and four methods, a cell that size is expected.

## 6. Conclusion

Duration does not modulate the collapse. This is consistent with, and
independent of, the selection study: the loss depends neither on *which* tokens
survive nor on *when* they occur — only on *how many*. Below the density
threshold, discarding costs ~15 points on this backbone regardless.

FlashVID shows no duration effect either, on either half, at any cut — it is
accuracy-neutral on LLaVA-OV throughout.

### On question-level spans

MotionBench's `qa[].start`/`end` fields do not support a separate
question-duration analysis. Only 2,530 of 8,052 rows carry them, 2,389 of those
equal the full video duration to within 0.1 s, and the longest span (95 s)
exceeds the longest video (49.96 s). The field is redundant where it is present
and inconsistent where it is not; video duration is the usable measure.

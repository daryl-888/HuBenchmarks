# Why the same numbers repeated across tables on re-runs

Verified at the PREDICTION level (not just the accuracy figure) on 2026-07-23.
There are **two different causes**, and they mean opposite things.

---

## Cause 1 — The same run, recorded under several names (BENIGN)

These are genuine duplicates: the *identical* model+config was executed more than
once (or the same output was copied into new dirs during the ovqwen→stage
restructure) and each copy got its own row in a table.

Proof — 0 of 8052 predictions differ:

| Score | Runs | Prediction diff |
|---|---|---|
| 2144/4018 = 53.36% | `ovqwen_dycoke_run1`, `dycoke_ovqwen15_fresh`, `dycoke_ovqwen2_fresh` | **0/8052** |
| 2132/4018 = 53.06% | `mdp3_qwen15_run`, `mdp3_ovqwen15_fresh`, `ovqwen2_mdp3_run1` | **0/8052** |
| 2135/4018 = 53.14% | `holitom_run1`, `ovqwen_holitom_run` | identical score |
| 2124/4018 = 52.86% | `ovqwen_videoitg_run2`, `videoitg_qwen2_infer` | identical score |

**Why it happened:** the project used to track "ovqwen15" and "ovqwen2" as if they
were two different backbones. They are the SAME model (`llava-ov-7b`, arch
`LlavaQwenForCausalLM`, Qwen2 internally) — the duplicate `llava-ov-7b-qwen2`
weight dir was a byte-for-byte copy, deleted 2026-07-23. `qwen_1_5` vs `qwen_2`
only changes prompt formatting. So "re-running on the other backbone" re-ran the
exact same thing and produced bit-identical output.

Deterministic decoding (`do_sample=False, temperature=0`) means the same model +
same config + same data ⇒ byte-identical predictions every time. Identical numbers
across re-runs are EXPECTED here, not a bug.

## Cause 2 — Different methods collapsing to the SAME number (NOT benign)

| Score | Runs | Meaning |
|---|---|---|
| 2116/4018 = 52.66% | `fastv_run1`, `fastv_ovqwen15`, `ovqwen_prunevid_run2` | **silent no-op** |

`ovqwen_prunevid_run2` differs from `fastv_run1` in **0/8052** predictions.
FastV's `apply_fastv()` was a stub (`enabled: false`) and PruneVID's VTP port
never fired on LLaVA-OV — so BOTH were the plain backbone wearing a method name.
Two "different methods" agreeing to the last sample is the signature of neither
method running. Both relabelled INERT; FastV has since been genuinely ported and
now diverges from baseline.

## A third, separate case — same score, different runs

`flashvid_run4` scores 2144/4018, the same as DyCoke, but differs in **1240/8052**
predictions. Same accuracy, different behaviour — coincidence, not duplication.
This is why score-matching alone is not evidence; only prediction-level comparison
settles it.

---

## The rule this establishes

Identical accuracy means nothing on its own. Always compare predictions:

```bash
python3 scripts/check_run.py <run> --expect-method <m> --vs-baseline <baseline>
```

* 0/N differ vs **another copy of the same config** → benign duplicate, dedupe the table.
* 0/N differ vs **the plain backbone** → silent no-op, the method never ran.
* Same score but N>0 differ → genuinely different runs that happen to tie.

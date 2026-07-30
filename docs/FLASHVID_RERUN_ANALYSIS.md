# FlashVID re-runs — why two pairs differ by 2.14% and 0.05%

Reconstructed from git history (`memory-bank/claude/progress.md` at commits
`e7c7e2e`, `c996a98`, `d877ab6`) plus the surviving run directories on Carya.

**Headline correction:** the two gaps have completely different causes, and
neither is what the labels suggest. The **2.14%** gap is **not** a prompt-template
effect — the two templates emit *byte-identical* prompts. It was a known bug. The
**0.05%** gap is a real, tiny retention effect.

---

## 1. The runs

| # | Run dir | Job | Retention (recorded) | Retention (**effective**) | Accuracy | Status |
|:-:|---|---|:---:|:---:|:---:|---|
| 1 | `flashvid_run4` | 7713093 | 0.25 | **0.25** | **53.36%** | ✅ valid |
| 2 | `s1_flashvid_r25_run` | 7933873 | 0.25 | **0.25** | **53.36%** | ✅ valid (2026-07-30 sweep) |
| 3 | `w2_flashvid_run` | — | 0.15 | **0.15** | **53.31%** | ✅ valid (gated reference) |
| 4 | `flashvid_qwen2_v2` | 7750906 | 0.25 ⚠️ | **0.15** | **53.29%** | ⚠️ record disagrees with execution — §5 |
| — | *"FlashVID (qwen15)"* | 7751030 | 0.15 | **0.10** ❌ | **51.22%** | ❌ **invalid** — wrong model class |

Run 5's data no longer exists on disk; it survives only in git history and in the
invalid-entries table of `master-results.md`.

---

## 2. The 2.14% gap — a bug, not a prompt difference

53.36% (run 1) vs 51.22% (run 5). The label said `qwen_1_5` vs `qwen_2`, which
implies a conversation-template difference. **It is not.** Rendering both
templates through the real `conv_templates` gives the same string, byte for byte:

```
qwen_1_5:  <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
           <|im_start|>user\n<image>\n{question}\nAnswer with the option's
           letter from the given choices directly.<|im_end|>\n
           <|im_start|>assistant\n

qwen_2:    ← identical, character for character
```

Same system message (`You are a helpful assistant.`), same roles
(`<|im_start|>user` / `<|im_start|>assistant`), same separator (`<|im_end|>`),
same `POST_PROMPT`. Under greedy decoding an identical prompt on identical
weights **must** give an identical answer, so the prompt cannot explain a 2.14%
gap.

**The real cause**, recorded in `master-results.md` §"Invalid entries": that run
*"ran at retention 0.10 with the wrong model class"* — the MotionBench variant
script loaded `LlavaLlamaForCausalLM` instead of the LLaVA-OneVision class, and
retention was **0.10**, not the 0.15 its filename claimed. Two independent
misconfigurations, neither related to the prompt.

> **The lesson:** a run's *label* is not evidence of its *configuration*. The
> `qwen15` in the name described a template that turned out to be irrelevant,
> while the two things that actually mattered — model class and retention — were
> both wrong and neither appeared in the name.

---

## 3. The 0.05% gap — a genuine retention effect

53.36% (run 1, r=0.25) vs 53.31% (run 3, r=0.15).

| | value |
|---|---|
| Accuracy difference | **0.05 pts** (2 of 4,018 questions) |
| Predictions differing | **1,123 / 8,052** (13.9%) |
| Resolution floor | ±1.54 pts |

**1,123 predictions changed to move the score by 0.05 points.** The retention
change had a large effect on *behaviour* and almost none on *accuracy* — the
gains and losses cancelled. This is the clearest illustration in the project of
why score-matching is never evidence: two runs can agree to within 0.05 points
while disagreeing on one answer in seven.

---

## 4. A true re-run reproduction

Runs 1 and 2 are the same configuration executed weeks apart, on different job
IDs, after a full repo restructure:

| | |
|---|---|
| Accuracy difference | **0.00 pts** |
| Predictions differing | **13 / 8,052** (0.16%) |

This is what determinism buys. Greedy decoding (`do_sample=False`) makes a re-run
reproducible to within the handful of samples affected by the known bad-NFS
videos. It is also the property the whole verification gate rests on: because a
faithful re-run diverges by ~13, a run diverging by **0** from the plain backbone
is proof the method never engaged.

---

## 5. `flashvid_qwen2_v2` executed at 0.15, not the 0.25 it records

**Correction history, stated plainly:** I first called this run mislabelled, then
retracted that because git showed the pre-fix code hardcoding `0.25` in both the
execution and the summary. The retraction was wrong. I trusted git to reflect the
*deployed* state after having already proven, in this same file, that it does
not (§5a). The behavioural evidence is decisive and I should have weighted it
higher.

### The divergence matrix settles it

Group all ten pairwise divergences by the retention each run *behaves* as:

| Retention gap | Divergences observed | Spread |
|---|---|:---:|
| 0.25 vs 0.25 | 13 | — |
| 0.15 vs 0.15 | 23 | — |
| **0.15 vs 0.25** | **1,114 · 1,118 · 1,123 · 1,127** | **13** |
| 0.10 vs 0.15 | 869 · 876 | 7 |
| 0.10 vs 0.25 | 1,506 · 1,508 | 2 |

**Four independent run-pairs at a 0.15↔0.25 gap agree to within 13 out of
~1,120 — a 1% spread.** Two independent pairs at 0.10↔0.25 agree to within 2.
A retention gap produces a reproducible, characteristic divergence, and every
pair lands in its bucket.

That partition only works if `flashvid_qwen2_v2` is assigned **0.15**. Assign it
the 0.25 it records and the matrix breaks: it would have to diverge by ~1,120
from `w2_flashvid_run` (it diverges by **23**) and by ~13 from `flashvid_run4`
(it diverges by **1,114**).

### So the numbers mean

| Run | Recorded | **Executed** | Basis |
|---|:---:|:---:|---|
| `flashvid_run4` | 0.25 | **0.25** | matches `s1_flashvid_r25_run` to 13 |
| `flashvid_qwen2_v2` | 0.25 ⚠️ | **0.15** | matches `w2_flashvid_run` to 23 |
| `w2_flashvid_run` | 0.15 | **0.15** | post-fix, CLI-driven, records `enabled` |

### How the record came to disagree with the execution

The fix commit `26c8ce4` changed **two** lines together — the `apply_flashvid`
call and the summary dict. On the deployed Carya copy those were evidently *not*
in sync on 2026-07-22: execution already took the CLI value (0.15) while the
summary still emitted the hardcoded `0.25`.

That is the same deploy-ahead-of-commit pattern independently established for
`attn_implementation`: the `_v2` log shows sdpa in use on 07-22, but the commit
adding sdpa is dated 07-23. The working copy on the cluster ran ahead of git in
both cases.

### What this does and does not explain

* The **1,114** gap between runs 1 and 4 is a **retention effect** (0.25 vs
  0.15) — *not* the attention implementation. My §5 elimination table reached
  the wrong conclusion there; the divergence matrix supersedes it.
* Attention and code-version differences account for the **small residuals**:
  13 between two identical-config runs, 23 between two same-retention runs from
  different script versions. Retention operates at the ~1,100 scale; code
  differences at the ~10–20 scale.

> **The methodological point survives, and is sharpened.** `summary.json` did not
> capture everything that determined the output — but the paired-divergence
> structure recovered the true configuration anyway. Behaviour was a more reliable
> witness than either the stored parameter or the commit history.

---

## 6. Subcategory breakdown

% correct within each category, over 4,018 scoreable questions.

| Run | Retention | Overall | Action Order | Camera Motion | Location | Motion Recog. | Motion Objects | Repetition |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `flashvid_run4` | 0.25 | 53.36% | 42.2 | 47.5 | 55.1 | 57.6 | 71.7 | 23.8 |
| `s1_flashvid_r25_run` | 0.25 | 53.36% | 42.2 | 47.5 | 55.1 | 57.5 | 71.9 | 23.8 |
| `w2_flashvid_run` | 0.15 | 53.31% | 39.9 | 45.7 | 53.8 | 58.0 | 71.7 | 28.2 |
| `flashvid_qwen2_v2` | 0.15 (records 0.25 ⚠️) | 53.29% | 39.7 | 45.7 | 54.0 | 57.8 | 71.9 | 28.2 |
| `s1_flashvid_r10_run` | 0.10 | 52.51% | 41.0 | 46.8 | 53.3 | 56.3 | 70.7 | 26.5 |
| *baseline* | — | *52.66%* | *40.5* | *45.2* | *55.5* | *57.0* | *71.2* | *23.8* |

**The subcategories cluster into exactly two groups**, and the clustering does
*not* follow the recorded retention:

* **Group A** — Action Order **42.2**, Camera Motion **47.5**, Repetition **23.8**
  → `flashvid_run4`, `s1_flashvid_r25_run` (both recorded 0.25)
* **Group B** — Action Order **39.7–39.9**, Camera Motion **45.7**, Repetition **28.2**
  → `w2_flashvid_run` (0.15), `flashvid_qwen2_v2` (**recorded 0.25**)

`flashvid_qwen2_v2` sits in Group B despite recording the same retention as the
two Group A runs. Combined with §5 — 23 predictions differing from the 0.15 run,
1,114 from a same-retention 0.25 run — the subcategory fingerprint is a **more
reliable identifier of what actually executed than the stored parameter is**.

What it does *not* do is tell us *why*. It shows run 4 behaved like a 0.15 run;
it cannot show whether that came from a different retention, a different eval
script, or a different model class. That question is unresolvable from the
surviving artifacts.

Note the trade rather than a uniform gain: Group A vs Group B **adds ~2.3 points
on Action Order and ~1.8 on Camera Motion** but **loses ~4.4 on Repetition
Count**. The near-identical overall scores (53.36 vs 53.31) hide two genuinely
different behaviours.

---

## 7. Prompt used — identical for every run

All five runs used exactly the same prompt construction
(`stage1-llava-ov/flashvid-motionbenc/eval_flashvid.py:58-59`):

```python
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."
msg = DEFAULT_IMAGE_TOKEN + "\n" + question + POST_PROMPT
conv = conv_templates[conv_template].copy()
conv.append_message(conv.roles[0], msg)
conv.append_message(conv.roles[1], None)
```

Rendered (a real MotionBench item):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
What is the person in the video holding in their hand?
A. Screw
B. Fountain pen
C. Screwdriver
D. Wrench
Answer with the option's letter from the given choices directly.<|im_end|>
<|im_start|>assistant
```

Generation was greedy for all five (`do_sample=False, temperature=0,
max_new_tokens=16`). **The prompt is a controlled constant across every run in
this analysis** — it explains none of the observed differences.

---

## 8. Summary

| Gap | Runs | Real cause |
|---|---|---|
| **2.14%** | 53.36% vs 51.22% | ❌ **Bug**, not prompt: wrong model class + retention 0.10. The `qwen15` label was irrelevant — the templates are byte-identical |
| **0.05%** | 53.36% vs 53.31% | ✅ Real retention effect (0.25 vs 0.15) — but **1,123 predictions changed** to produce it |
| **0.00%** | run 1 vs run 2 | ✅ True re-run reproduction, 13/8,052 differ (bad-NFS videos) |
| **0.07%** | run 1 vs run 4 | ⚠️ A **retention** difference (0.25 vs 0.15) that `summary.json` hides — run 4 records 0.25 but executed 0.15 |

Three things worth carrying forward:

1. **Prompt differences explain none of it.** `qwen_1_5` and `qwen_2` are the same
   string on this backbone; any result attributed to that distinction should be
   re-examined.
2. **Accuracy hides behaviour.** A 0.05-point gap concealed 1,123 changed
   predictions, and a 0.00-point gap concealed only 13. The number alone
   distinguishes neither.
3. **The summary did not capture everything that mattered.** Runs 1 and 4 agree
   on every recorded field and still differ on 1,114 predictions — so provenance
   was incomplete, not merely mislabelled. This is why the current sweep verifies
   retention arithmetically from each run's ACTIVE log instead of trusting the
   stored parameter.

# Determinism, and Whether This Benchmark Is Sound

An honest assessment of *why re-running a model reproduces identical results*, what
that property legitimately proves, and — importantly — **where this methodology is
weak**. Written to be read critically, including by us.

---

## 1. Why identical inputs give byte-identical outputs

Every eval in this repo decodes greedily:

```python
model.generate(..., do_sample=False, temperature=0, max_new_tokens=16)
```

`do_sample=False` removes the only stochastic step. At each position the model
produces logits over the vocabulary and takes `argmax` — no sampling, no
temperature, no top-k/top-p. The chosen token is appended and the loop repeats.

Every other stage is fixed too:

| Stage | Why deterministic |
|---|---|
| Frame sampling | `np.linspace(0, total-1, n)` — fixed indices, never random |
| Vision encoding | frozen weights; `model.eval()` disables dropout |
| LLM forward | same weights + same inputs ⇒ same logits |
| Token choice | `argmax`, not a draw |

So the output is a **pure function** of (weights, frames, prompt). Re-running does
not produce "similar accuracy" — it reproduces the same 8,052 strings exactly.

### This is empirically confirmed, not just theory

Same configuration, run on different days, different compute nodes, across
intervening code changes:

| Comparison | Differing predictions |
|---|---|
| `ovqwen_dycoke_run1` vs `dycoke_ovqwen15_fresh` | **0 / 8052** |
| `ovqwen_dycoke_run1` vs `w2_dycoke_run` | **0 / 8052** |
| `mdp3_qwen15_run` vs `w2_mdp3_run` | **0 / 8052** |

---

## 2. What we legitimately get from this

Determinism converts "identical output" into **evidence**, and we used it twice:

**(a) Detecting duplicate table rows.** DyCoke's 53.36% appeared in three result
directories. 0/8052 differences proved they were one configuration recorded three
times (the "ovqwen15 vs ovqwen2" split that turned out to be the same model), not
three independent measurements. The table was double-counting.

**(b) Detecting silent no-ops — the higher-value use.** If a method truly prunes
85% of visual tokens, the model receives different inputs, so *some* outputs must
change. When the PruneVID→LLaVA-OV port produced **0/8052 differences from the
plain backbone**, that was proof the pruning never executed. The same check caught
FlashVID and AIM on Qwen3-VL, which printed `ACTIVE: keep=1750 (15.0%)` while
emitting byte-identical output — under determinism, that combination is only
possible if the computed mask was never applied. It wasn't.

Methods that *do* work are clearly separated from each other, not just from
baseline — evidence they are genuinely distinct computations:

| | vs baseline | dycoke | holitom | aim |
|---|---|---|---|---|
| dycoke | 1031 | — | 1496 | 1182 |
| holitom | 1838 | 1496 | — | 1464 |
| aim | 1406 | 1182 | 1464 | — |
| mdp3 | 1452 | 1172 | 1459 | 1160 |

---

## 3. Where this methodology is FLAWED

### 3.1 The headline accuracy differences are not statistically significant

This is the most important caveat in the whole project. DyCoke is our best
LLaVA-OV method at 53.36% vs a 52.66% baseline — a +0.70 point "win". Testing it
properly (McNemar's test on paired predictions, the correct test since both models
answer the *same* questions):

```
baseline wrong → dycoke right : 170
baseline right → dycoke wrong : 142
net gain                      : 28 questions  (+0.70 points)
McNemar χ² = 2.34             (needs ≥ 3.84 for p < 0.05)
```

**χ² = 2.34 ⇒ not significant.** The +0.70 is within what coin-flipping on 312
disagreements would produce. The same holds for FlashVID (+0.65), HoliTom (+0.48)
and MDP3 (+0.40) — all smaller than DyCoke's.

**Therefore: "DyCoke is the best method" is not a supportable claim.** The
defensible claim is that on LLaVA-OV these methods are **indistinguishable from the
backbone and from each other**. Any ranking among them is noise. This is why the
results table must not be read as a leaderboard.

By contrast, the ~10-point Qwen3-VL vs LLaVA-OV backbone gap is far outside noise
and *is* a real finding.

### 3.2 Determinism is conditional, and we did not test its limits

Bit-identical reproduction holds for **same hardware + same software + same batch
size**. It can break across:

* different GPU models (different kernels, different reduction orders),
* CUDA/PyTorch version changes,
* different batch sizes (changes accumulation order),
* non-deterministic kernels (some scatter/atomic ops).

Floating-point addition is not associative — `(a+b)+c ≠ a+(b+c)` — so a different
summation order can perturb a logit in its last bits. If two candidate tokens are
nearly tied, `argmax` flips, and every subsequent token can change.

Our comparisons were all same-cluster, same-env, batch size 1, so this did not
bite. **But we never verified reproduction on a different GPU generation**, so we
cannot claim hardware-independent reproducibility. A replicator on different
hardware should expect *close* numbers, not identical ones.

### 3.3 Zero divergence does not strictly prove "no-op"

`0/8052` proves the method did not change the **outputs**. In principle a method
could prune only tokens the model was already ignoring, execute perfectly, and
still produce identical answers.

We mitigate this by requiring **two independent signals** before recording a
number: the method's own `ACTIVE` log (the mechanism ran, with token counts) **and**
prediction divergence (it affected the output). Neither alone suffices — that is
exactly the FlashVID/AIM lesson, where the first was present and the second absent.
The residual risk is a method that is genuinely correct *and* genuinely
output-neutral; we have not observed one, but the gate would mislabel it.

### 3.4 Standardized retention is not paper-faithful for every method

We force **15% retention** for cross-method comparability. FastV's paper default is
`r=0.5` (keep 50%); at 15% it drops to 36.78%. That number is a valid measurement
**of FastV at 15%**, not of FastV as published. The table says so, but it is easy
to misread. Methods without a single retention knob (DyCoke `l/p/k`, STTM
threshold, MDP3, AIM, VideoITG) keep paper defaults, so the comparison across
methods is **not** at equal compute.

### 3.5 Single-seed, single-configuration

Because decoding is deterministic, we have exactly **one** measurement per cell.
There is no variance estimate from repeated runs — re-running gives the identical
number by construction. Our error bars come from the *binomial* nature of the
score (≈±1.5 points at n=4018, 95% CI), not from run-to-run spread. Any claim
smaller than that interval is unsupported.

### 3.6 Other limitations

* **MotionBench only** — no claim generalizes to other video benchmarks.
* **Two backbones** — n=2 is not a trend.
* **50% of items are `NA`** and excluded; scoring depends on that protocol choice.
* **Letter-matching** (`\b([A-D])\b`) can mis-score verbose answers, though
  spot-checks showed it handling Qwen3-VL's "C. Screwdriver" style correctly.

---

## 4. Verdict

**Determinism is a genuine and valuable property here** — it is what makes the
silent-no-op gate possible, and that gate caught four real failures that would
otherwise have entered the results as legitimate numbers (FastV stub, PruneVID-OV,
VisionZip empty-output, FlashVID/AIM mask-dropped).

**But determinism is a verification tool, not a substitute for statistics.** It
tells you *whether a method did anything*. It says nothing about whether the
resulting accuracy difference is meaningful — and by McNemar's test, on LLaVA-OV
**none of our method-vs-baseline differences are**.

### How the results should be stated

* ✅ "The backbone dominates: Qwen3-VL-8B (62.52%) vs LLaVA-OV-7B (52.66%)."
* ✅ "On LLaVA-OV, all tested efficiency methods perform within noise of the
  backbone; at matched 15% retention, FastV and VisionZip degrade substantially."
* ✅ "Every reported number is verified to have actually engaged its method."
* ❌ "DyCoke is the best method." — **not supported** (χ²=2.34).
* ❌ "Method A beats method B by 0.2 points." — **noise**.

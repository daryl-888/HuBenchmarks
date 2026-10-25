# Research Contributions

## Research Question

**Does DyCoke's token compression hurt a model's ability to understand motion?**

Motion understanding is particularly sensitive to temporal information. DyCoke prunes tokens across
frames, which could throw away exactly the tokens that capture motion between frames. This has not
been tested on MotionBench specifically — making this evaluation a new contribution.

---

## Why This Matters

DyCoke achieves 1.5x inference speedup and 1.4x memory reduction by merging similar tokens across
frames (Stage 1) and dropping low-attention tokens from the KV cache (Stage 2). The open question
is whether these pruned tokens carry motion-critical information. MotionBench is the right benchmark
to answer this because it is specifically designed to test fine-grained motion understanding —
something general VQA benchmarks don't capture.

---

## Three Contributions

### 1. LLaVA-OV-7B Baseline on MotionBench
This number does not exist publicly. Establishes where LLaVA-OneVision-7B sits on MotionBench
without any compression. Serves as the reference point for evaluating DyCoke's impact.

### 2. LLaVA-OV-7B + DyCoke on MotionBench
Also not publicly available. Shows what happens to accuracy when DyCoke's temporal token merging
(k=0.3) and KV cache pruning (p=0.8) are applied. First published result of DyCoke on a
motion-specific benchmark.

### 3. Per-Category Analysis
The most informative part. DyCoke may affect some motion categories more than others:

| Category | Why it might be sensitive to pruning |
|----------|--------------------------------------|
| Repetition Count | Requires tracking repeated motion across many frames — most likely to be hurt by token merging |
| Action Order | Needs temporal ordering of events — pruning similar tokens could collapse distinct actions |
| Motion Recognition | General motion type — may be robust since the signal is strong |
| Location-related Motion | Spatial + temporal — moderate sensitivity |
| Camera Motion | Global signal across all frames — likely most robust to pruning |
| Motion-related Objects | Object-level motion — depends on whether object tokens get pruned |

---

## Context: Known MotionBench Results

For comparison, published results from the official leaderboard (motion-bench.github.io):

| Model | Overall Accuracy |
|-------|-----------------|
| Gemini-3-Pro-Preview | 70.4% |
| Gemini-2.5-Pro | 66.3% |
| GPT-5.1 | 66.2% |
| TE Fusion (SOTA method) | 58.0% |
| Most VLMs | below 60% |

LLaVA-OV-7B is not on this leaderboard — your run establishes that number.

---

## What the Results Will Tell Us

- **If DyCoke accuracy ≈ baseline:** Token compression does not hurt motion understanding.
  DyCoke is safe to use for motion-related tasks, and the speedup comes for free.

- **If DyCoke drops significantly overall:** The pruned tokens carry motion-critical information.
  DyCoke's parameters may need tuning (lower k, lower p) for motion tasks specifically.

- **If DyCoke drops in specific categories only:** Reveals exactly where temporal token merging
  is harmful — e.g., if Repetition Count drops but Camera Motion holds, that confirms pruning
  hurts fine-grained temporal tracking more than global motion signals.

---

## Run Configuration

| Setting | Baseline | DyCoke |
|---------|----------|--------|
| Model | LLaVA-OV-7B | LLaVA-OV-7B |
| dycoke | False | True |
| dycoke_l | N/A | 3 |
| dycoke_p | N/A | 0.8 |
| dycoke_k | N/A | 0.3 |
| Frames | 32 | 32 |
| Resolution | 384×384 | 384×384 |
| GPU | 1× Ada GPU (Carya) | 1× Ada GPU (Carya) |
| Samples | 8052 | 8052 |
| Output | results/baseline_run1 | results/dycoke_run1 |

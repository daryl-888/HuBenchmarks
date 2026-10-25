# HoliTom × MotionBench Evaluation

Evaluates HoliTom — a holistic token merging method combining pre-LLM and intra-LLM compression — on the MotionBench benchmark using LLaVA-OneVision-7B on the UH Carya HPC cluster.

---

## What is HoliTom?

HoliTom (NeurIPS 2025) explicitly connects outer-LLM and inner-LLM compression, arguing that either side alone leaves efficiency on the table.

- **Outer stage:** Global redundancy-aware temporal segmentation + spatiotemporal merging before the LLM
- **Inner stage:** Token-similarity-based merging inside the LLM
- **Key value:** Holistic design reduces visual sequences before the LLM, then continues reducing residual redundancy inside the model
- Training-free

Category: Input & model-wise hybrid compression | Training-free

---

## Why HoliTom × MotionBench?

HoliTom's dual-stage compression is aggressive — it prunes both before and inside the LLM. MotionBench tests whether this double reduction discards motion-critical tokens that a single-stage method might retain.

---

## Repository Structure

```
holitom-motionbenc/
├── run_holitom.sbatch       # Main SLURM job — runs HoliTom inference on MotionBench
├── run_baseline.sbatch      # Baseline run (no HoliTom)
├── test_holitom.sbatch      # 10-sample sanity check
├── cache_dataset.sbatch     # One-time dataset cache (already done via dycoke run)
└── tasks/
    └── motionbench/
        ├── motionbench.yaml
        └── utils.py
```

---

## Key Paths on Carya

| Path | Purpose |
|------|---------|
| `/project/rhu/dpalfaro/code/HoliTom` | HoliTom source code |
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | LLaVA-OneVision-7B weights |
| `/project/rhu/dpalfaro/conda/envs/dycoke11` | Conda environment |
| `/project/rhu/dpalfaro/results/holitom_run1/` | Output directory |

---

## Paper

- NeurIPS 2025
- Code: Official GitHub (see literature doc)

# PruneVid × MotionBench Evaluation

Evaluates PruneVid — a training-free video token pruning method combining redundancy reduction with question-aware pruning — on the MotionBench benchmark using LLaVA-OneVision-7B on the UH Carya HPC cluster.

---

## What is PruneVid?

PruneVid (Findings of ACL 2025) is a training-free visual-token pruning method for efficient multimodal video understanding.

- **Stage 1:** Reduces intrinsic video redundancy via spatial-temporal token merging
- **Stage 2:** Leverages LLM reasoning and question-to-video attention to keep question-relevant visual features
- **Key value:** Bridge between input-side redundancy compression and model-side, query-aware visual-token pruning
- Training-free

Category: Input & model-wise hybrid compression | Training-free

---

## Why PruneVid × MotionBench?

PruneVid's question-aware pruning stage should theoretically preserve motion-relevant tokens when the question asks about motion. MotionBench directly tests this: does question conditioning help retain motion information?

---

## Repository Structure

```
prunevid-motionbenc/
├── run_prunevid.sbatch      # Main SLURM job — runs PruneVid inference on MotionBench
├── run_baseline.sbatch      # Baseline run (no PruneVid)
├── test_prunevid.sbatch     # 10-sample sanity check
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
| `/project/rhu/dpalfaro/code/PruneVid` | PruneVid source code |
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | LLaVA-OneVision-7B weights |
| `/project/rhu/dpalfaro/conda/envs/dycoke11` | Conda environment |
| `/project/rhu/dpalfaro/results/prunevid_run1/` | Output directory |

---

## Paper

- Findings of ACL 2025
- Code: Official GitHub (see literature doc)

# VideoITG × MotionBench Evaluation

Evaluates VideoITG — an instruction-guided frame selection method for VideoLLMs — on the MotionBench benchmark using LLaVA-OneVision-7B on the UH Carya HPC cluster.

---

## What is VideoITG?

VideoITG (CVPR 2026 Highlight) reframes frame sampling as instructed temporal grounding: frames selected for a video should depend on the user instruction, not only on generic redundancy.

- **VidThinker:** Automatic annotation pipeline generating instruction-conditioned clip captions
- **VideoITG-40K:** Dataset produced by VidThinker for training the frame selector
- **Plug-and-play selector:** Improves downstream VideoLLMs by aligning frame selection with semantic instruction following and temporal grounding

Category: Input-side / instruction-guided frame selection | Trained (not purely training-free)

---

## Why VideoITG × MotionBench?

MotionBench questions are motion-specific — asking about action order, camera motion, repetition count, etc. VideoITG's instruction-conditioned frame selection should theoretically select frames most relevant to motion queries. This evaluation tests whether that holds.

---

## Repository Structure

```
videoitg-motionbenc/
├── run_videoitg.sbatch      # Main SLURM job — runs VideoITG inference on MotionBench
├── run_baseline.sbatch      # Baseline run (no VideoITG)
├── test_videoitg.sbatch     # 10-sample sanity check
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
| `/project/rhu/dpalfaro/code/VideoITG` | VideoITG source code |
| `/project/rhu/dpalfaro/weights/llava-ov-7b` | LLaVA-OneVision-7B weights |
| `/project/rhu/dpalfaro/weights/videoitg-selector` | VideoITG frame selector weights (TODO) |
| `/project/rhu/dpalfaro/conda/envs/dycoke11` | Conda environment (may need new env) |
| `/project/rhu/dpalfaro/results/videoitg_run1/` | Output directory |

---

## Paper

- CVPR 2026 Highlight
- Code: Official GitHub (see literature doc)

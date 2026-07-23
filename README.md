# HuBenchMarks

Benchmarking efficient video understanding LLMs on **MotionBench** (8,052 samples, ~4,018 scoreable) on the UH Carya HPC cluster.

## Structure

```
llava-ov-7b/          LLaVA-OV-7B (Qwen 1.5 backbone) — 8 models
llava-ov-7b-qwen2/    LLaVA-OV-7B-Qwen2 (Qwen2 backbone) — 8 models
other-backbones/      Non-LLaVA-OV models (STTM, DyTo, PruneVid, VisionZip, etc.)
memory-bank/          Project documentation and context
patches/              Source-level patches for model repos
analysis/             Results tables and analysis scripts
```

## Quick Reference

- **Full docs**: See [CLAUDE.md](CLAUDE.md) — rules, cluster paths, setup notes, results
- **Current status**: See [memory-bank/activeContext.md](memory-bank/activeContext.md)
- **Benchmark details**: See [BENCHMARKING_NOTES.md](BENCHMARKING_NOTES.md)

## Results Summary

| Model | Backbone | Accuracy | Status |
|-------|----------|:--------:|:------:|
| STTM-LLaVAVid | LLaVA-Video-7B | 54.28% | ✅ |
| HoliTom | LLaVA-OV-7B | 53.11% | ✅ |
| VideoITG | LLaVA-OV-7B | 52.51% | ✅ |
| FlashVID | LLaVA-OV-7B-Qwen2 | 51.99% | ✅ |
| FastVID | LLaVA-OV-7B-Qwen2 | 51.92% | ✅ |
| MDP3 | LLaVA-OV-7B | 53.25%* | ⏳ partial |
| AIM | LLaVA-OV-7B-Qwen2 | — | 🔄 pending |
| DyTo | LLaVA-NeXT-Vicuna-7B | — | 🔄 pending |

Full results table in [analysis/MotionBench_Results.md](analysis/MotionBench_Results.md).

# Product Context — HuBenchmarks

## Why This Project Exists

Video LLMs are developing rapidly, but their efficiency-accuracy tradeoffs on real video benchmarks are poorly documented — especially for token/pruning methods that claim to reduce compute while maintaining accuracy. This project provides a structured, reproducible evaluation of 11+ efficient video understanding models against a single benchmark (MotionBench) on a shared hardware platform (UH Carya HPC).

## Problems It Solves

1. **Cross-model comparison confusion**: Each model paper reports results on different datasets, different frame counts, different backbone LLMs. This project evaluates them all on the same 8,052-sample benchmark.
2. **Reproducibility gap**: Many models have fragile setup requirements (specific torch/transformers version pins, permission-constrained conda envs, missing dependencies). The sbatch files and setup docs capture the exact environment needed.
3. **NFS stale-video fragility**: MotionBench's video dataset sits on shared NFS with stale file handles that D-state the kernel. A subprocess-isolated `load_video` patch (black frames on timeout) is required to keep jobs running.

## How It Should Work

1. User runs `sbatch test_<model>.sbatch` (smoke test, `--limit 50`)
2. If job fails → check traceback → fix → sync to Carya → resubmit
3. If job succeeds → submit `run_<model>.sbatch` (full eval, no limit)
4. Record accuracy in the Results Summary table in CLAUDE.md

## User Experience Goals

- **Single-command submission**: Each model has exactly two sbatch files (test + run)
- **Clear error diagnosis**: Tracebacks from SLURM's `.err` files should map directly to the known-fixed-issues list
- **No surprises from stale files**: Sync protocol (heredoc → verify with grep) prevents the "fixed locally but not on Carya" bug

# HuVLLM — Efficient Video-LLM Methods on MotionBench

Benchmarking training-free efficiency methods (visual-token pruning / merging /
frame selection) for video LLMs on **[MotionBench](https://motion-bench.github.io/)**,
across two backbones a year and a half apart:

| Backbone | Era | Baseline |
|---|---|:---:|
| **LLaVA-OV-7B** | mid-2024 | 52.66% |
| **Qwen3-VL-8B** | late-2025 | **62.52%** |

**Headline finding: the backbone dominates the method.** The ~10-point jump from
upgrading the backbone dwarfs every efficiency method — on LLaVA-OV, all methods
land within ~0.8 points of the baseline (within noise).

## 📖 Documentation → [`docs/`](docs/)

| Start here | |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | **The results table** — every method × backbone, with per-category breakdowns |
| [docs/SETUP.md](docs/SETUP.md) | Weights, dataset, conda envs, and the config to edit for your system |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | How runs are scored and **verified to actually engage** |
| [docs/backbones/](docs/backbones/) | Per-backbone pages (architecture, methods, baselines) |
| [docs/methods/](docs/methods/) | Per-method pages (algorithm, our port, params, reproduce command) |

## Methods (11)

DyCoke · FlashVID · HoliTom · MDP3 · AIM · VideoITG · STTM · FastV · VisionZip ·
PruneVID · DyTo — each faithful to its paper, and each **verified to change model
behavior** rather than silently degrading to the backbone (a real, recurring
failure mode — see [Methodology](docs/METHODOLOGY.md)).

## Repository layout

```
config/            paths.sh — EDIT to replicate on your system
docs/              replication hub (start here)
stage1-llava-ov/   methods on LLaVA-OV-7B      (one dir per method)
stage3-qwen3-vl/   methods on Qwen3-VL-8B      (real ports + PORT_FEASIBILITY.md)
other-backbones/   methods on their native backbones (PruneVID/PLLaVA, VisionZip/LLaVA-1.5, DyTo/Vicuna, STTM/LLaVA-Video)
scripts/           check_run.py (verification gate), deploy.sh, launch helpers
patches/           source diffs applied to the authors' method repos
memory-bank/       full working record incl. master-results.md
archive/           descoped / superseded material, kept for provenance
```

Each method dir has: `eval_<method>.py` (the runner), `.sbatch` files (our exact
cluster submissions), `tasks/motionbench/` (dataset config), and a README pointing
to its `docs/` page.

## Reproduce one run

```bash
source config/paths.local.sh   # your edited copy of config/paths.sh
python stage1-llava-ov/dycoke-motionbenc/eval_dycoke.py \
    --model_path "$W_LLAVA_OV" --meta_path "$MOTIONBENCH_META" \
    --output_dir "$HUVLLM_RESULTS/dycoke_run" --num_frames 32 --conv_template qwen_2
python scripts/check_run.py "$HUVLLM_RESULTS/dycoke_run" \
    --expect-method dycoke --vs-baseline "$HUVLLM_RESULTS/baseline_run"
```

## Status

LLaVA-OV: all methods gated ✅. Qwen3-VL: baseline done, method ports verifying 🔄.
Live status in [docs/RESULTS.md](docs/RESULTS.md).

## Notes

- Runs were executed on the UH Carya SLURM cluster; the `.sbatch` files reflect
  that. `config/paths.sh` + the per-method `python` command let you run without SLURM.
- Method source repos are the original authors' code, cloned and patched
  ([patches/](patches/)); please cite the original papers (linked per method).
- License: see [LICENSE](LICENSE).

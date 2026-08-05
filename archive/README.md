# archive/

Material kept for provenance but **not** part of the active benchmark. Nothing here
is needed to replicate our results — it documents what was tried and set aside.

| Folder | What | Why archived |
|---|---|---|
| `stage2-llava-video-descoped/` | The Stage-2 tree (LLaVA-Video-7B backbone) — method dirs + sbatch. | Stage 2 was planned as a third backbone but **descoped mid-project**; its jobs were cancelled and it has **no gated results**. Kept so the attempted scope is on record. Note: `sttm-llavavid` (STTM on LLaVA-Video-7B, 53.33%) is a *separate* real result and lives under `other-backbones/`, not here. |
| `legacy-sbatch/` | Pre-restructure sbatch templates (`sbatch-files/`). | Written before the `stage{1,2,3}/` reorganization; they reference old flat paths (`/code/fastv-motionbenc/`) that no longer hold current code. Superseded by the per-method sbatch inside each stage dir. |
| `deprecated-stage3-stubs/` | The original 7 Stage-3 `eval_<method>.py` "ports". | Every one was the **same Qwen3-VL baseline applying no compression** — running one produced the plain backbone number wearing a method's name. Replaced by real `eval_<method>_qwen3vl.py` implementations. (A one-line loud-fail guard remains in place at the live path so an old job script fails instead of silently faking a result.) |
| `nested-dycoke_ex/` | Duplicate `eval_dycoke.py` copies that sat in a nested `dycoke_ex/` subdir. | Redundant with the canonical `eval_dycoke.py`. |

See `../docs/` for the current, replicable structure.

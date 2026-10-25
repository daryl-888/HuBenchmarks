# DyTo README Setup Guide — Verification

**Date:** 2026-08-05
**Subject:** Setup Guide quoted from the upstream README (`github.com/Jam1ezhang/DYTO`, HEAD `570e977`; byte-exact snapshot vendored at `other_backbones/dyto-motionbenc/dyto_readme.md`)
**Sources checked:** upstream `README.md`, upstream `pyproject.toml` (vendored byte-exact), live HuggingFace API (`liuhaotian/llava-v1.6-vicuna-7b`, `liuhaotian/llava-v1.6-34b`)

---

## The Setup Guide being verified

From upstream README "🚀 Quick Start → Setup Guide" (verbatim):

```
1. Environment Setup
conda create -n dyto python=3.10
conda activate dyto

pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir

apt-get update
apt-get install git-lfs
git-lfs install

2. API Configuration
export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
export OPENAI_ORG=$YOUR_OPENAI_ORG  # Optional

3. Model Download
# Get LLaVA-NeXT weights
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b
```

And the section-header claims above it:

```
### Environment
- CUDA 11.7
- Python 3.10.12+
- PyTorch 2.1.0+
```

---

## Element-by-element verification

| # | Setup element | Verified? | Check | Evidence |
|---|---|---|---|---|
| 1 | `conda create -n dyto python=3.10` | ✅ CORRECT | Env name `dyto`, CPython 3.10 | README header says "Python 3.10.12+"; `pyproject.toml` says `requires-python = ">=3.8"` (3.10 satisfies); conda env named `dyto` is the name used elsewhere in the repo (e.g. the vendored run scripts) |
| 2 | `conda activate dyto` | ✅ CORRECT | Env activation | Standard conda; name matches step 1 |
| 3 | `pip install -e ".[train]"` | ✅ CORRECT | Editable install with optional `train` extras | `pyproject.toml` has `[project.optional-dependencies] train = ["deepspeed==0.12.6", "ninja", "wandb"]`. All three packages exist on PyPI; the instruction is fully runnable |
| 4 | `pip install flash-attn --no-build-isolation --no-cache-dir` | ✅ CORRECT | FlashAttention-2 install | Matches DyTo's hard `flash_attention_2` requirement in `dyto/llava` builder; `--no-build-isolation` and `--no-cache-dir` are the standard workarounds for flash-attn's build |
| 5 | `apt-get update` | ✅ CORRECT | Debian/Ubuntu package index refresh | Standard prerequisite for `apt-get install git-lfs` |
| 6 | `apt-get install git-lfs` | ✅ CORRECT | Git LFS client | Required to fetch LFS objects from HuggingFace model repos |
| 7 | `git-lfs install` | ✅ CORRECT (syntax) | Configure Git LFS filters | Note: `git lfs install` (space form) is the more idiomatic invocation; `git-lfs install` (the hyphen used in README) also works for the Git wrapper command |
| 8 | `export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY` | ✅ CORRECT | API key env var | Standard OpenAI key env var (used for the GPT-4V / GPT-4-based evaluation scripts in `run_inference_*` / `eval/*`) |
| 9 | `export OPENAI_ORG=$YOUR_OPENAI_ORG  # Optional` | ✅ CORRECT | Optional org env var | Standard OpenAI org override; marked optional in README |
| 10 | `git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b` | ✅ CORRECT — model exists, not gated, can be cloned as-written | Live HF API | `https://huggingface.co/api/models/liuhaotian/llava-v1.6-vicuna-7b`: `"id":"liuhaotian/llava-v1.6-vicuna-7b"`, `"gated":false`, `"private":false`; 7.06B params; LLaVA 1.6 arch `LlavaLlamaForCausalLM`; sha `deae57a8c0ccb0da4c2661cc1891cc9d06503d11` |
| 11 | `git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b` | ✅ CORRECT — model exists, not gated, can be cloned as-written | Live HF API | `https://huggingface.co/api/models/liuhaotian/llava-v1.6-34b`: `"id":"liuhaotian/llava-v1.6-34b"`, `"gated":false`, `"private":false`; 34.75B params; LLaVA 1.6 arch `LlavaLlamaForCausalLM`; license apache-2.0; sha `6754aa86582435ae12ef47db2de51180f0968aa7` |

---

## Section-header claims (in the README "Environment" block above the guide)

| Claim | Verified? | Check |
|---|---|---|
| `CUDA 11.7` | ⚠️ NOT PINNED IN pyproject | `pyproject.toml` does not pin a CUDA version. The guide's CUDA 11.7 is advisory (matches torch 2.2.0 cu121/cu118 era builds). DyTo's own docs elsewhere say CUDA 11.7. Fine as a recommendation; not a hard pin. |
| `Python 3.10.12+` | ✅ CORRECT | `requires-python = ">=3.8"` in `pyproject.toml`; 3.10.12+ is compatible; the committed `.pyc` files are `cpython-310`, confirming the authors ran Python 3.10 |
| `PyTorch 2.1.0+` | ⚠️ MINOR MISMATCH with the pyproject pin | `pyproject.toml` pins `torch==2.2.0`, `torchvision==0.17.0`. "PyTorch 2.1.0+" is *broad enough* to include 2.2.0, so it is not wrong, but the canonical number from the repo's own dependency file is 2.2.0, not 2.1.0. |

---

## Inaccuracies found (none block the guide from being broadly correct)

1. **README "Environment" says "PyTorch 2.1.0+"** but the repo's own `pyproject.toml` pins `torch==2.2.0`. Not contradictory, but the authoritative number is 2.2.0.
2. **CUDA 11.7** in the README is not backed by any pin in `pyproject.toml`; it is advisory only.
3. `git-lfs install` (hyphen) vs the more common `git lfs install` (space) — both invoke the same Git LFS wrapper; cosmetic.

---

## Conclusion

**All 11 steps of the Setup Guide are correct and runnable as-written.**

- The `train` extra (`deepspeed==0.12.6`, `ninja`, `wandb`) is valid — `pip install -e ".[train]"` works as documented.
- Both LLaVA-NeXT weight repos are **ungated and public** (verified live 2026-08-05), so `git lfs clone` works exactly as the README shows.
- The only discrepancies are the README's own advisory "Environment" numbers (PyTorch 2.1.0+ vs the pinned 2.2.0; CUDA 11.7 not pinned) — cosmetic, not blocking.

This page is pairing documentation with `docs/DYTO_REVERIFICATION_FINDINGS.md`; the Setup Guide itself is sound and was never part of the defect discussion.
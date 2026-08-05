# Environment lockfiles

`pip freeze` captured from the **exact conda environments that produced the
published results** on UH Carya (2026-07-24). Each file is a full transitive pin,
so a replicator does not have to reverse-engineer versions from prose.

## Which env does a method need?

| Env | Key pins | Methods |
|---|---|---|
| `dycoke11` | transformers **4.40.0**, torch 2.12.0+cu130 | DyCoke, FastV, AIM, FlashVID, STTM, VisionZip, VideoITG (LLaVA-OV) |
| `qwen3vl` | **transformers 5.14.1**, torch 2.6.0+cu124 | **all Qwen3-VL work** |
| `holitom` | transformers **4.45.2** (exact) | HoliTom |
| `mdp3` | + DPP/vlmeval deps | MDP3 |
| `prunevid` | — | PruneVID (PLLaVA) |
| `dyto` | transformers **4.40.0**, torch 2.12.0+cu130 | DyTo (Vicuna) |
| `videoitg`, `visionzip`, `flashvid`, `aim`, `sttm_new`, `fastvid` | — | as named |

## Use

```bash
conda create -n <env> python=3.10 -y
conda activate <env>
pip install -r config/envs/<env>.txt
```

## Caveats

* These are **pip** pins. Conda-level and system CUDA are not captured; see
  `docs/SETUP.md` for the CUDA/driver context.
* Some entries are cluster-local editable installs of the method source repos —
  clone those from the URLs in `docs/SETUP.md` and apply `patches/` instead.
* The envs deliberately hold **conflicting** `transformers` versions (4.38 / 4.45 /
  5.14). That is not an accident: Qwen3-VL requires 5.x, several LLaVA methods
  break above 4.45. Do not attempt to unify them.

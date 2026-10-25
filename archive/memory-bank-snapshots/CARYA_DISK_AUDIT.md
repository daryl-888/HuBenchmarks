# Carya Disk Audit — 2026-07-23

Filesystem `/project/rhu` = 1.0T shared, was **97% full (34G free)**.

## Reclaimed this pass: +15G (34G → 49G free)

| Removed | Size | Why it was safe |
|---|---:|---|
| `cache/http-v2` | **8.7G** | Stale HTTP download cache, last written Jun 18. Regenerated on demand; nothing reads it directly. |
| `conda/envs/fastv` | **6.3G** | Documented broken torch (`register_fake` missing). Every working FastV path uses `dycoke11`; the 8 sbatch files that still named it were repointed first, so refs = 0 before deletion. |
| `conda/envs/dyto-v2` | 6.3G | (earlier) unreferenced duplicate of `dyto` |
| 24 empty result dirs + 203 stale logs | ~small | (earlier) no summary.json, no rows |

## Remaining big consumers — reviewed, NOT removed

| Path | Size | Verdict |
|---|---:|---|
| `weights/` | **~103G** | All 7 in active use. `llava-video-7b` (15G) is Stage-2 only, which is descoped — **the single largest reclaimable item if Stage 2 stays cancelled.** |
| `conda/envs/` | **~74G** | 13 envs, all referenced by live sbatch except as noted. `ppgdata` (7.6G) appears unrelated to this project — needs the owner's OK. |
| `cache/huggingface/hub` | 4.9G | **Do not delete.** `siglip-so400m` (3.3G) and `clip-vit-large` (1.6G) hold REAL weights loaded at runtime (SigLIP is LLaVA-OV's vision tower). The other 5 entries are 4KB pointers. |
| `code/` | 940M | Active source trees. |
| `mdp3_pkgs` + `aim_pkgs` | 3.2G | `pip --target` installs both still referenced by live sbatch PYTHONPATHs. |
| `results/` | 213M | Small. Not the problem — cleaning results cannot fix disk pressure. |

## Candidates needing your decision

1. ~~`weights/llava-video-7b` (15G)~~ — **DO NOT DELETE.** Initially flagged as
   Stage-2-only, but verification shows it is the backbone for `sttm-llavavid` in
   other-backbones, which holds a real 53.33% result. Still required.
2. **`conda/envs/ppgdata` (7.6G)** — owned by dpalfaro (created Jun 24) — referenced by nothing in this repo and unrelated
   to the benchmark. May belong to another project; confirm before removing.
3. ~~`conda/envs/{trajvit,imove}`~~ — verified: these envs **do not exist**. The 4
   sbatch files referencing them are dead templates for methods with no public code.

## Rule learned
`results/` and log cleanup are cosmetic here (213M + 29M). The real levers are
**weights** and **conda envs** — together ~177G of the 990G used.

# patches/

Reconstruct the **exact** method source trees that produced the published results.

Two pieces work together:

1. **Pinned upstream commits** — `config/paths.sh` exports `$SHA_DYCOKE`,
   `$SHA_HOLITOM`, etc. These are the revisions we actually ran. If an upstream
   repo force-pushes or rewrites history, only these SHAs recover our code.
2. **Real unified diffs** — `diffs/*.patch`, generated with `git diff` from the
   working trees (bytecode/egg-info excluded). They apply with `git apply`.

## Use

```bash
source config/paths.sh          # or your edited paths.local.sh
patches/apply_all.sh --check    # dry run: verify SHAs + patch applicability
patches/apply_all.sh            # check out the pins and apply the patches
```

`--check` is idempotent and safe; the script also detects patches that are
*already* applied and reports them rather than failing.

## What is patched

| Repo | Pinned SHA | Patch | What it changes |
|---|---|---|---|
| DyCoke | `dd7463498203` | `DyCoke.patch` (87 ln) | NFS-safe `load_video` (subprocess + hard-kill; stale handles hang the kernel in D-state) |
| HoliTom | `e9b2972f6895` | `HoliTom.patch` (140 ln) | transformers-4.45 compat stubs, SigLIP `device_map` fix, guarded `llava` import |
| PruneVid | `b12600c6176c` | `PruneVid.patch` (104 ln) | `mlp_bias` guard in `llama.py`, VTP shape fix |
| MDP3 | `45616806d117` | — | unmodified upstream |
| VisionZip | `8f86b55c6f00` | — | unmodified upstream |
| FastV | `f95102a10acf` | — | unmodified upstream |

**DyTo** is the exception: our copy has **no git history**, so it cannot be pinned
by SHA. It is used as vendored files at `$SRC_DYTO` (upstream:
`github.com/Yunkang-Sun/DyTo`). This is a known reproducibility gap — see
`docs/VALIDITY_ASSESSMENT.md`.

## Legacy

The `dyto/`, `holitom/`, `prunevid/` subdirectories hold **whole-file copies** from
before we generated diffs. They are superseded by `diffs/` and kept only for
reference.

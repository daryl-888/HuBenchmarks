# DyTo Upstream Re-verification — Consensus Findings

**Date:** 2026-08-05
**Subject:** `github.com/Jam1ezhang/DYTO` (ICCV 2025, "Beyond Training: Dynamic Token Merging for Zero-Shot Video Understanding")
**Pinned HEAD:** `570e977` (2025-10-25, "Update README.md")
**Status:** Re-verified against the live upstream git history and the committed bytecode.

---

## Executive verdict

**The released DyTo source is genuinely incorrect — it cannot be run end-to-end
from any published revision using any official config — but the committed
bytecode proves the authors ran a *working* private tree. This is a
release-hygiene / dirty-commit failure, not a deliberate misstatement.**

The paper's numbers were produced with a private working tree that included a
returning `finch_clusterv2`, a FINCH build with the `tw_finch` parameter, and a
`scripts/` runner directory. None of those were shipped in the repository.

---

## The five findings

### Finding 1 — `finch_cluster()` has no `return` statement (blocking)

`dyto/llava/model/llava_arch.py:188–218`:

```python
def finch_cluster(self,image_features,tw_finch):
    T, N, D = image_features.shape
    image = image_features[:,0,:].detach().cpu().numpy() #[num_frame,577,4096]

    c, num_clust, _ = FINCH(image,verbose=False,tw_finch=tw_finch)   # :192

    classification = torch.tensor(c[:, np.argsort(num_clust)[-2] if len(num_clust) > 1 else -1])
    unique_classes = torch.unique(classification, return_inverse=False, return_counts=False)
    new_embeddings = []
    selected_indexs = []
    if len(unique_classes) <= 6:
        num = 12 // len(unique_classes)
        for class_label in unique_classes:
            class_indices = (classification == class_label).nonzero(as_tuple=True)[0]
            num_to_select = min(len(class_indices), num)
            selected_indexs.extend(class_indices[:num_to_select].tolist())
    else:
        for class_label in unique_classes:
            class_indices = (classification == class_label).nonzero(as_tuple=True)[0]
            selected_index = class_indices[len(class_indices)//2 if len(class_indices) > 2 else -1]
            selected_indexs.append(selected_index)
    # <-- function body ENDS here. No return. Returns None unconditionally.
```

The caller at lines 231–234 immediately dereferences the result:

```python
elif temporal_aggregation == "spatial_tome_finch_dynamic_all_frms":
    # FINCH
    clus_image_features = self.finch_cluster(image_features,tw_finch = True)
    if clus_image_features.shape[0] >25:      # <-- AttributeError: 'NoneType'
```

**Impact:** Every run using the official `TEMPORAL_AGGREGATION` value crashes
with `'NoneType' object has no attribute 'shape'`.

---

### Finding 2 — the call site has never matched a defined, returning function in ANY published revision

The complete git history of `llava_arch.py` is two commits:

| Commit | Call site at `temporal_aggregation(...)` | Result |
|---|---|---|
| `2c6d0f7` ("Initial commit") | `self.finch_clusterv2(image_features, tw_finch=True)` | `finch_clusterv2` **does not exist** in any revision's source → `NameError` |
| `ae46634` ("commit for arch") | `self.finch_cluster(image_features, tw_finch=True)` | Renamed to the return-less `finch_cluster` → `NoneType` crash |

The diff between the two commits is exactly this rename (plus comment
translation). No revision ever defined a working function behind the call.

---

### Finding 3 — the committed `.pyc` is the smoking gun (authors ran a working `finch_clusterv2`)

The repository accidentally commits `__pycache__` artifacts (sloppy `.gitignore`).
Among them:

```
dyto/llava/model/__pycache__/llava_arch.cpython-310.pyc
```

Extracting the bytecode (CPython 3.10) from that file shows the code object:

```
<code object ... qualname='LlavaMetaForCausalLM.finch_clusterv2'>
```

and the disassembly ends with a **`RETURN_VALUE`** opcode after building
`new_embeddings` / `torch.stack(...)` — i.e. the `finch_clusterv2` the authors
actually compiled and ran **did return a tensor**. The source committed to git is
a different, incomplete edit of the same file.

**Conclusion:** the working code that produced the paper's numbers existed, ran,
and left bytecode behind — but was never committed as source.

---

### Finding 4 — the official config references files that were never published

`cfgs/Baseline/dyto_llava_7b-resize-100frms_spatial_tome_finch_dynamic_all_frms.yaml`
(the primary 7B config):

```yaml
SCRIPT: [
  "bash scripts/run_eval_videoqabench.sh",            # Openset VideoQA tasks
  "bash scripts/run_eval_multiplechoiceqabench.sh",   # Multiple Choice VideoQA tasks
  "bash scripts/run_eval_generativebench.sh",         # Text Generation tasks
]
...
TEMPORAL_AGGREGATION: "spatial_tome_finch_dynamic_all_frms"   # the broken path
```

A full `git ls-tree -r --name-only HEAD` shows **no `scripts/` directory exists
anywhere in the repository**. The runner scripts (which would also reveal the
exact command-line invocation used for the published numbers) were never shipped.

Additionally, the broken `spatial_tome_finch_dynamic_all_frms` aggregation is the
value in the official config — it is the **intended primary code path** of DyTo,
not an obscure variant.

---

### Finding 5 — the pinned dependency cannot satisfy the code's call

`pyproject.toml` pins `finch-clust==0.2.0` (an exact PyPI version, not a fork).
Pristine PyPI 0.2.0's `FINCH()` signature is:

```python
FINCH(data, initial_rank, req_clust, distance, ensure_early_exit, verbose, use_ann_above_samples)
```

`tw_finch` occurs **zero** times in the published library. The code calls:

```python
FINCH(image, verbose=False, tw_finch=tw_finch)
```

TW-FINCH is a separate published algorithm (Sarfraz et al., CVPR 2021) and a
separate implementation in the `ssarfraz/FINCH-Clustering` repo — not a parameter
of the base `FINCH()`. The authors developed against a local FINCH modification
that was never published or referenced.

---

## Consensus

| Question | Answer |
|---|---|
| Is the released code incorrect? | **Yes** — non-runnable end-to-end from any commit, using any official config. |
| Was it intentional? | **No** — the committed `.pyc` proves a working private tree produced the results. |
| What actually happened? | The authors committed a **stale/partial working tree**: a renamed call site (`finch_clusterv2` → `finch_cluster`), a truncated paste in `finch_cluster()`, a local unpinned FINCH build, and an unpublished `scripts/` runner. |
| Is this a criticism of the research? | No. The defects are release-artifact engineering problems — the paper's method is not in question. |

This finding **supports** `docs/UPSTREAM_DEFECTS.md` §1.1/§1.2 and adds the
mechanism:

> **The authors committed a different (older) revision of `llava_arch.py` than
> the one that produced their numbers.**

---

## Reproducibility guidance

- `docs/UPSTREAM_DEFECTS.md` — original defect analysis (§1.1 `tw_finch`,
  §1.2 missing return) and the transcription/reconstruction methods used.
- `patches/dyto/finch_cluster_return.patch` — the 8-line return tail
  transcribed verbatim from the sibling `kmeans_cluster()` in the same file.
- `analysis/upstream-faithful/dyto-official/` — unmodified-upstream verification
  run (wrapper-only; no DyTo source edits) demonstrating the committed code
  cannot run as-is.
- `results-cache/ob_dyto_run/summary.json` — the gated result (42.06%),
  labeled `"DyTo (reconstructed TW-FINCH)"`, `paper_faithful: false`.
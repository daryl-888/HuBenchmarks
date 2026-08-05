# Upstream cross-reference: validity and accuracy audit

Every method checked against its **live GitHub repository** (July 2026), not
against our local clone or our own prior notes. Three classes of problem were
found: wrong citation metadata, backbone claims that do not match released
code, and configurations that match no published setting.

Clone provenance was read from each repo on Carya
(`git -C <repo> remote -v && git log -1`).

---

## 1. Citation metadata — 8 of 11 were wrong

| Method | Recorded here | Correct | Source |
|---|---|---|---|
| DyCoke | arXiv 2024, `2411.14401` | **CVPR 2025**, arXiv **2411.15024** | [KD-TAO/DyCoke](https://github.com/KD-TAO/DyCoke) |
| FastV | arXiv 2024 | **ECCV 2024 Oral** | [pkunlp-icler/FastV](https://github.com/pkunlp-icler/FastV) |
| PruneVID | — 2024 | **ACL 2025** | [visual-ai/prunevid](https://github.com/visual-ai/prunevid) |
| HoliTom | — 2025 | **NeurIPS 2025** | [cokeshao/HoliTom](https://github.com/cokeshao/HoliTom) |
| STTM | — 2025 | **ICCV 2025** | [HYUNJS/STTM](https://github.com/HYUNJS/STTM) |
| VideoITG | — 2025 | **CVPR 2026 Highlight** | [NVlabs/VideoITG](https://github.com/NVlabs/VideoITG) |
| VisionZip | — 2024 | **CVPR 2025** | [dvlab-research/VisionZip](https://github.com/dvlab-research/VisionZip) |
| MDP3 | **ICCV 2025** | **arXiv only**, `2501.02885` | [sunh-23/MDP3](https://github.com/sunh-23/MDP3) |
| AIM | ICCV 2025 | ICCV 2025 — correct | [LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM) |
| FlashVID | ICLR 2026 Oral | correct | [Fanziyang-v/FlashVID](https://github.com/Fanziyang-v/FlashVID) |
| DyTo | ICCV 2025 | correct, arXiv `2411.14401` | [Jam1ezhang/DYTO](https://github.com/Jam1ezhang/DYTO) |

Two are worth singling out:

- **DyCoke was cited with DyTo's arXiv ID.** `2411.14401` is *Beyond Training:
  Dynamic Token Merging* (DyTo). DyCoke is `2411.15024`. Both appear in this
  project, so the collision propagated silently.
- **MDP3 was promoted to ICCV 2025 without a source.** It is an arXiv preprint
  (January 2025) with no venue on its repo or listing. Claiming a venue it does
  not have is the most serious of these errors and is now corrected to arXiv.

## 2. Backbone support vs released code

| Method | Backbones the paper/repo claims | Ours | Verdict |
|---|---|---|---|
| PruneVID | PLLaVA, ST-LLM, **LLaVA-OneVision** | PLLaVA + our LLaVA-OV port | **released code ships only `models/pllava`** |
| VideoITG | InternVL2.5/3.5, Qwen3-VL, LLaVA-Video, Eagle2.5 | LLaVA-OV + Qwen3-VL | LLaVA-OV is **not** a supported backbone |
| VisionZip | LLaVA-1.5, Qwen2.5-VL | LLaVA-1.5 + Qwen3-VL partial | consistent; LLaVA-OV correctly excluded |
| STTM | LLaVA-Video-7B/72B, LLaVA-OneVision-7B, Qwen2VL-7B | all three of ours | supported |
| FlashVID | LLaVA-OV, LLaVA-Video, Qwen2.5-VL, **Qwen3-VL** | LLaVA-OV + our Qwen3-VL port | **upstream already supports Qwen3-VL** |
| HoliTom | LLaVA-OneVision | LLaVA-OV + Qwen3-VL port | supported |
| DyCoke | LLaVA-OV (+ community Gemma3, Qwen2.5-VL) | LLaVA-OV + Qwen3-VL port | supported |

Three consequences:

**PruneVID's LLaVA-OneVision results are not reproducible from released code.**
The repo contains `models/pllava` and nothing else; the commit history ends at
`b12600c` (2025-05-15) with no OneVision work at any point. Our LLaVA-OV number
(38.20%) is therefore a reimplementation with **no upstream reference to check
against** — which is precisely why it cannot be cited as PruneVID's LLaVA-OV
performance. Its PLLaVA number (44.13%) is the reproducible one.

**VideoITG on LLaVA-OV is off-spec.** The authors never claim that backbone. Our
52.86% is a port to an unsupported model and should be labelled as such.

**Our Qwen3-VL FlashVID port is not FlashVID.** The repo ships
`flashvid/modeling_qwen3_vl.py` and `scripts/qwen3_vl.sh`. Our port never
imports the package — it hand-implements a scoring rule and omits, on Qwen3-VL:
inner-LLM compression entirely (`pruning_layer=28`, `llm_retention_ratio=0.1`),
`expansion=1.25`, `min_segment_num=4`, `complementary_segment`,
`segment_threshold` and `token_selection_method=attn_div`. The 56.65% cell is
our approximation of the method, not the method.

The authors'-code re-run surfaced a second, independent problem: their
`flashvid` package hard-requires FlashAttention-2
(`assert self.config._attn_implementation == "flash_attention_2"` in
`modeling_qwen3_vl.py`'s vision attention, no sdpa/eager fallback), which was
not installed in the `qwen3vl` env, and separately their patched
`Qwen3VLVisionModel_forward` drops the `.to(hidden_states.dtype)` cast that
stock `transformers` applies to interpolated position embeddings — a real
regression in the released code, not an environment gap. Status and fix in
progress; see [FLASHVID_OFFICIAL_RERUN.md](FLASHVID_OFFICIAL_RERUN.md).

## 3. Configurations that match no published setting

### STTM — the published configs are per-benchmark *and* per-budget

From `scripts/eval/run_vidqa.sh`:

```
sttm_llava_common_cfg   = --sa_start_layer_idx 2 --sa_tree_root_level 1
sttm_qwen2vl_common_cfg = --sa_start_layer_idx 2 --sa_tree_root_level 2
sttm_cfg_50_vnb_llavavideo_7b  = ... --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.65
sttm_cfg_30_vmme_llavavideo_7b = ... --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.50
```

| Our run | thresh / temporal / root | Status |
|---|---|---|
| LLaVA-OV | 0.85 / 0.65 / 1 | matches the published `_50_vnb` LLaVA pairing |
| LLaVA-Video | 0.80 / 0.65 / 1 | **mixes** the 50% (0.85/0.65) and 30% (0.80/0.60) configs |
| Qwen3-VL | 0.85 / **−1.0** / **0** | **no published config uses `root_level=0`**; LLaVA uses 1, Qwen2VL uses 2. `temporal_thresh=−1.0` disables temporal merging entirely |

The Qwen3-VL cell is the problem: it runs a spatial-only variant at a tree root
level the authors never use. It should not be compared against the LLaVA-OV
STTM cell, and the earlier note that STTM "has no published operating point" was
wrong — it has many, keyed to benchmark and budget.

### FastV — non-canonical clone; the authors *do* state a video setting

We cloned `chenllliang/FastV` (the first author's personal copy) at `f95102a`,
**2024-03-20**. The canonical repo is `pkunlp-icler/FastV` (ECCV 2024 Oral),
pulled here at `d165972`, 2025-01-04.

**Correction to an earlier claim in this file.** It previously said our R=85%
was "outside the official repo's demonstrated range". That was wrong — it came
from a README summary rather than the scripts. Both official eval scripts sweep
`rank_list=(72 144 288 432)` against 576 image tokens, i.e. keep 12.5–75%
(R = 87.5% down to 25%). **Our R=85% (keep 15%) sits inside that range.**

Note the two official scripts disagree in their own comments about the mapping:
`eval_ocrvqa_*` annotates the list `R=(75% 50% 25% 12.5%)` while
`eval_aokvqa_*` annotates it `R=(87.5% 75% 50% 25%)`. The second is the
arithmetically correct one — `rank = (1-R)·576`, so rank 72 is R=87.5%.

What the official repo *does* pin down:

- **`Ks=(2)`** in both eval scripts; the README's results tables report K=2 at
  R=25/50/75% and one K=3/R=50% FLOPs comparison.
- **A stated video recommendation.** The README says that "in contexts
  requiring video understanding, which involves around ten times image tokens,
  the latency reduction achieved by fastv (with **K=2 and R=50%**) can reach up
  to 25% without hurting the performance." That is the authors' own video
  setting, and their "ten times image tokens" is exactly our regime
  (6,273 vs 576).
- **Still no video evaluation in either repo.** The official one ships OCR-VQA,
  AOKVQA latency, and the LLaVA-1.5 image suite (GQA, SEED, MME, MMBench,
  VizWiz, SQA). The video claim is made in prose, not in runnable code.

So the accurate statement is: our keep-15% is a normal *fraction* for FastV, but
the authors' stated video setting is **K=2, R=50%**. Now run
(`s1_fastv_r50_run`, and `s1_fastv_k3r50_run` for the README's K=3/R=50% row):
**36.34% and 35.94%**, statistically indistinguishable from keep-15%'s 36.78%.
The published operating point does not recover the collapse — see
[FASTV_COLLAPSE_ANALYSIS.md](FASTV_COLLAPSE_ANALYSIS.md) §1.

### AIM — faithful run, wrong recorded metadata

Our `llava_arch.py` on Carya is **byte-identical** to
[LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM) at `edc24e9` (verified by
`diff`). The run is faithful. Two problems follow from that, not from it:

- Upstream ships **two** active merge steps — `r=orig_num//2` then
  `r=orig_num//4`, i.e. **25% retention**. The `//8`, `//16`, `//32`, `//64`
  lines are commented out.
- Our `summary.json` records `"merge": "bipartite_soft_matching, 4 steps
  (50%/25%/12.5%/6.25%)"`. That string is **wrong**: only two steps ran.

Consequence: **AIM's LLaVA-OV cell is at 25% retention, not the standardized
15%**, so it is not budget-comparable with the other LLaVA-OV rows, and the
appendix budget table that lists AIM at 15% is wrong for this backbone. The
accuracy figure itself (52.86%) is unaffected — the run did what upstream does.

AIM's published eval also requires `attn_implementation=eager`; ours sets it, so
that is correct. It uses `conv_template=qwen_1_5` against our `qwen_2`, which
render identical prompts on this build.

## 4. Clone freshness

| Repo | Our HEAD | Date | Note |
|---|---|---|---|
| FastV | `f95102a` | 2024-03-20 | **2.3 years stale, and the non-canonical repo** |
| PruneVid | `b12600c` | 2025-05-15 | current upstream HEAD |
| MDP3 | `4561680` | 2025-07-14 | |
| VisionZip | `8f86b55` | 2025-07-21 | |
| HoliTom | `e9b2972` | 2025-10-10 | |
| DYTO | `570e977` | 2025-10-25 | |
| FastVID | `a40a109` | 2025-11-10 | |
| DyCoke | `dd74634` | 2025-11-22 | |
| STTM | `336bf36` | 2026-01-25 | |
| VideoITG | `50a60a8` | 2026-04-17 | |
| FlashVID | `983cce6` | 2026-05-01 | |
| AIM | `edc24e9` (upstream) | 2025-10-09 | Carya clone is another user's and `git` refuses it, but its `llava_arch.py` **diffs clean against upstream** |
| FastV (official) | `d165972` | 2025-01-04 | pulled for this audit into gitignored `upstream-refs/` |

## 5. What this changes

**Corrected, no re-run needed.** All venue and arXiv metadata; the STTM
"no published operating point" claim; PruneVID's backbone attribution; the
claim that FastV's R=85% was outside the official sweep (it is inside); AIM's
recorded merge-step count.

**Labelling changes.** PruneVID-LLaVA-OV and VideoITG-LLaVA-OV must be labelled
ports to unsupported/unreleased configurations, not method results. **AIM on
LLaVA-OV must be labelled 25% retention**, not 15% — it is not budget-comparable
with the other rows on that backbone.

**Re-run, resolved.**

| Run dir | Result | Consequence |
|---|---|---|
| `s3_sttm_pub_run` | 61.60%, −0.92 vs Qwen3-VL backbone, not significant | replaces the off-spec `root_level=0 / -1.0` cell (−5.45); STTM is accuracy-neutral on Qwen3-VL at a published setting, matching its LLaVA-OV result |
| `s1_fastv_r50_run`, `s1_fastv_k3r50_run` | 36.34%, 35.94% | published K=2/R=50% and K=3/R=50% land in the same collapse band as keep-15%; see [FASTV_COLLAPSE_ANALYSIS.md](FASTV_COLLAPSE_ANALYSIS.md) |
| `s1_fastv_r75_run` | 35.69% | keep-75% also collapses; the flat range now spans keep 10–75% |
| `s1_prunevid_c100_run` | 53.36%, +0.70, not significant | `cluster_ratio=1.0` recovers, matching FastV's own `keepall` control — same density-threshold cause |
| `s3_flashvid_official_run` | in progress | blocked twice: a dtype crash in the authors' code (fixed locally) then a hard FlashAttention-2 requirement with no matching prebuilt wheel on this cluster (building from source). See [FLASHVID_OFFICIAL_RERUN.md](FLASHVID_OFFICIAL_RERUN.md). |

**Unresolvable.** PruneVID on LLaVA-OneVision has no released reference
implementation, so our port cannot be validated against anything. AIM's clone
provenance cannot be established without fixing repo ownership on Carya.

---

Sources: [DyCoke](https://github.com/KD-TAO/DyCoke) ·
[FlashVID](https://github.com/Fanziyang-v/FlashVID) ·
[HoliTom](https://github.com/cokeshao/HoliTom) ·
[MDP3](https://github.com/sunh-23/MDP3) ·
[AIM](https://github.com/LaVi-Lab/AIM) ·
[VideoITG](https://github.com/NVlabs/VideoITG) ·
[STTM](https://github.com/HYUNJS/STTM) ·
[FastV](https://github.com/pkunlp-icler/FastV) ·
[PruneVid](https://github.com/visual-ai/prunevid) ·
[VisionZip](https://github.com/dvlab-research/VisionZip) ·
[DYTO](https://github.com/Jam1ezhang/DYTO) ·
[arXiv 2411.15024](https://arxiv.org/abs/2411.15024) ·
[arXiv 2501.02885](https://arxiv.org/abs/2501.02885)

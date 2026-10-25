# What's Left — 2026-07-23 end of session

Target: **11 methods × 2 backbones (LLaVA-OV-7B, Qwen3-VL-8B) = 22 cells**,
32 frames, 0.15 retention where the method exposes one, paper-exact.

---

## ✅ DONE

**Verification harness** — `scripts/check_run.py` (+`--smoke`, `--vs-baseline`),
`scripts/deploy.sh`, gates wired into 13 sbatch files, every method's summary
emits `"enabled": true`.

**10 of 11 LLaVA-OV methods smoke-verified as actually engaging:**
FastV, DyCoke, HoliTom, AIM, STTM, PruneVID, MDP3, VisionZip, FlashVID
(+ VideoITG with a warning: its infer script emits no `*_params` block).

**Real bugs fixed today** (each was a silent or fast failure, none was a
benchmarking problem):
| Bug | Method(s) |
|---|---|
| `apply_fastv()` was a stub; rebuilt paper-exact from the authors' code | FastV |
| eager attention ⇒ 100% empty generations | FastV |
| `output_ids[:, input_ids.shape[1]:]` discarded the whole response | VisionZip |
| builder defaulted to flash_attn (not installed) ⇒ ImportError | DyCoke, FlashVID, VideoITG |
| loaded `LlavaLlamaForCausalLM` (wrong class) at retention 0.10 | FlashVID |
| `libnccl.so.2` was in the conda env, not `mdp3_pkgs` | MDP3 |
| wrong arg names (`--pool-frames`, `--frame-counts`) | MDP3, FlashVID |
| called `eval_videoitg.py` instead of `..._infer.py` | VideoITG |
| Qwen3-VL processor re-sampled frames, ignoring `--num_frames` | Qwen3-VL |
| pip installs landed in `~/.local`, invisible under `PYTHONNOUSERSITE=1` | qwen3vl env |
| 16 sbatch files corrupted by the restructure (mid-line `\`) | repo-wide |
| 137 sbatch job names normalized; two files shared a log name | repo-wide |
| post-commit hook pushed `main` from `restructure`, failing every commit | repo |

**Qwen3-VL unblocked** — no env could load `Qwen3VLForConditionalGeneration`
(all had transformers 4.45). New `qwen3vl` env works; full baseline running.

---

## 🔄 IN FLIGHT
* **7770357** — Qwen3-VL full baseline, 8,052 samples, ~8h. This is Stage-3 Step 0:
  a real result AND the `--vs-baseline` reference every Qwen3-VL port needs.

---

## ❌ WHAT'S LEFT

### 1. DyTo — never ran (only unverified method)
5 consecutive import failures: `cannot import name 'LlavaLlamaForCausalLM' from
'dyto.llava.model'`. Its one "result" (5.25%) emitted free-form captions instead
of answer letters — below the 25% random floor. **Not a result.** Needs the
import fixed, or documentation as unrunnable.

### 2. Wave 2 — LLaVA-OV full runs (11 cells)
**Every existing full-run number predates today's fixes and is provisional.**
Re-run the 10 verified methods at 32f/0.15, then gate each with
`--vs-baseline`. ~5-7h per run. Highest-value: FastV (first real result),
VisionZip (all prior runs were the empty-output bug), FlashVID (prior runs used
retention 0.10 + wrong model class).

### 3. Wave 3 — Qwen3-VL ports (11 cells) — the big one
**All 8 existing Stage-3 "ports" are the same baseline script applying no method.**
Each needs genuine re-implementation against a non-LLaVA architecture:
1. FastV (algorithm already resolved on LLaVA-OV)
2. DyCoke, HoliTom
3. VisionZip, AIM, MDP3, VideoITG, FlashVID
4. PruneVID, STTM, DyTo — likely **architecture-incompatible** (STTM patches Qwen2
   attention specifically; DyTo is Vicuna-bound). Document as holes, never fake.

### 4. Housekeeping
* VideoITG infer script needs a `*_params` block (gate can't confirm engagement).
* Dedupe the table: DyCoke 53.36% appears in 3 dirs, MDP3 53.06% in 3 — same run,
  0/8052 predictions differ (see DUPLICATE_RESULTS_EXPLAINED.md).
* **Disk 97% full (35G free)**, `weights/` = 100G. Clear space before Wave 2+3.

---

## Realistic scope
22 cells total: **0 fully gated full runs yet**, 10 methods verified ready on
LLaVA-OV, 1 backbone baseline in flight, 11 Qwen3-VL cells requiring real
implementation work. The LLaVA-OV half is now a compute problem; the Qwen3-VL half
is still an engineering problem.

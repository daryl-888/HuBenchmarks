# MotionBench Results

## Overall Accuracy

| # | Model | Accuracy | Correct/Scoreable | NA Skipped | Backbone (vLLM) | Key Parameters | Status |
|:-:|-------|:-------:|:-----------------:|:----------:|:----------------|:---------------|:------:|
| 1 | **STTM-LLaVAVid** | **54.28%** | 2,181 / 4,018 | 4,034 | LLaVA-Video-7B-Qwen2 | sa_start=2, tree_thresh=0.85, temp_thresh=0.65 | ✅ |
| 2 | **MDP3** | **53.25%** | 1,100 / 2,066 | 2,037 | LLaVA-OneVision-7B | — | ⏳ *(50% data)* |
| 3 | **HoliTom** | **53.11%** | 2,134 / 4,018 | 4,034 | LLaVA-OneVision-7B | retain_ratio=0.15, T=0.80, k=18, r=0.5 | ✅ |
| 4 | **VideoITG** | **52.51%** | 2,110 / 4,018 | 4,034 | LLaVA-OneVision-7B | Grounding-based | ✅ |
| 5 | **FlashVid** | **51.92%** | 2,086 / 4,018 | 4,034 | LLaVA-OV-7B-Qwen2 | retention=0.1, alpha=0.7, temp_thresh=0.8, 8 frames | ✅ |
| 6 | **STTM-v2** | **51.77%** | 2,080 / 4,018 | 4,034 | LLaVA-OneVision-7B | sa_start=2, tree_thresh=0.85, temp_thresh=0.65 | ✅ |
| 7 | **PruneVid** | **43.80%** | 1,760 / 4,018 | 4,034 | LLaVA-OneVision-7B | cluster_ratio=0.5, segment_ratio=0.25, layer=10, alpha=0.4, tau=0.8 | ✅ |
| 8 | **VisionZip** | **40.09%** | 1,611 / 4,018 | 4,034 | LLaVA-v1.5-7B | dominant=54, contextual=10, 8 frames | ✅ |
| 9 | **AIM** | — | — | — | LLaVA-Qwen2-7B | Token merging + PageRank pruning | 🔄 7662816 |
| 10 | **DyTo** | 3.70% | 1 / 27 | 23 | LLaVA-NeXT-Vicuna-7B | 100 frames, FINCH clustering | 🔄 7662788 |

## Per-Category Breakdown

| Model | Action Order | Camera Motion | Location-related Motion | Motion Recognition | Motion-related Objects | Repetition Count |
|-------|:-----------:|:-------------:|:----------------------:|:-----------------:|:---------------------:|:----------------:|
| **STTM-LLaVAVid** | 0.4104 | 0.4883 | **0.5751** | **0.5825** | **0.7174** | 0.2750 |
| **HoliTom** | 0.4085 | **0.4961** | 0.5256 | 0.5697 | 0.7145 | 0.2725 |
| **VideoITG** | 0.4027 | 0.4623 | 0.5330 | 0.5697 | 0.7014 | 0.2650 |
| **FlashVid** | 0.3950 | 0.4701 | 0.5513 | 0.5379 | 0.7130 | 0.2800 |
| **STTM-v2** | 0.3931 | 0.4805 | 0.5421 | 0.5365 | 0.7014 | **0.2950** |
| **PruneVid** | 0.3622 | 0.3377 | 0.3938 | 0.4628 | 0.6362 | 0.2600 |
| **VisionZip** | 0.3372 | 0.3299 | 0.3718 | 0.4107 | 0.5739 | 0.2575 |

## Active Jobs

| JobID | Model | State | Time | Notes |
|-------|-------|:-----:|:----:|-------|
| 7662191 | MDP3 full run | **RUNNING** | 4h on compute-10-4 | ~50% data processed |
| **7662816** | **AIM full run** | **PENDING** | — | boundaries=[] fix applied |
| 7662788 | DyTo full run | PENDING | — | waiting for GPU |

## Key Observations

- **STTM-LLaVAVid** leads overall (54.28%) and dominates in Location Motion, Motion Recognition, and Motion Objects categories
- **HoliTom** is strongest on Camera Motion (49.61%) — its dual-stage compression preserves camera motion well
- **STTM-v2** leads Repetition Count (29.50%) — temporal tree merging helps with counting tasks
- **PruneVid** and **VisionZip** underperform overall — their aggressive token pruning discards motion-critical information
- **DyTo** test (3.70%) used the base LLaVA-NeXT-Vicuna-7B model without fine-tuning; full run with actual weights is pending
- **AIM** was crashing at sample 542 due to `kwargs.get('boundaries', None)` returning `None` instead of a list; fixed to `[]`
- Results sorted by overall accuracy descending
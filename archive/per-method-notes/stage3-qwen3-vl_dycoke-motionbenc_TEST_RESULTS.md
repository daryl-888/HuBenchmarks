# Test Run Results (10 Samples)

**Model:** LLaVA-OV-7B + DyCoke (l=3, p=0.8, k=0.3)  
**Date:** 2026-05-27  
**Samples:** 10 (limit run for sanity check)  
**Overall Accuracy:** 2/5 = 40% (NA samples excluded per official MotionBench protocol)

---

## Per-Sample Results

| doc_id | Category | Target | Predicted | Correct |
|--------|----------|--------|-----------|---------|
| 0 | Action Order | C | B | ✗ |
| 1 | Motion-related Objects | C | C | ✓ |
| 2 | Motion Recognition | NA | A | — skipped |
| 3 | Motion Recognition | NA | C | — skipped |
| 4 | Repetition Count | D | B | ✗ |
| 5 | Location-related Motion | C | A | ✗ |
| 6 | Motion Recognition | NA | A | — skipped |
| 7 | Camera Motion | NA | A | — skipped |
| 8 | Motion-related Objects | C | C | ✓ |
| 9 | Action Order | NA | B | — skipped |

---

## Notes

- **5 out of 10 samples have target `NA`** — unanswerable questions skipped per official MotionBench eval protocol (`compute_accuracy.py` lines 29-30: `if qa["answer"] == "NA": continue`)
- Out of 8052 total samples, **4034 (50%) are NA** — only 4018 are scoreable
- The 2 correct answers were both `Motion-related Objects` category (doc_id 1 and 8)
- 40% on 5 samples is not representative — full run of 4018 answerable samples needed for meaningful results

---

## Per-Sample Detail

| doc_id | Category | Video | Target | Predicted | Correct |
|--------|----------|-------|--------|-----------|---------|
| 0 | Action Order | 37e1b635...mp4 | C | B | ✗ |
| 1 | Motion-related Objects | ef476626...mp4 | C | C | ✓ |
| 2 | Motion Recognition | 671023d2...mp4 | NA | A | — |
| 3 | Motion Recognition | Hn0cwNTG...mp4 | NA | C | — |
| 4 | Repetition Count | c916cd65...mp4 | D | B | ✗ |
| 5 | Location-related Motion | rRzcgJZp...mp4 | C | A | ✗ |
| 6 | Motion Recognition | 8t4poBBC...mp4 | NA | A | — |
| 7 | Camera Motion | 6p0CVyfn...mp4 | NA | A | — |
| 8 | Motion-related Objects | 4655b2b7...mp4 | C | C | ✓ |
| 9 | Action Order | 08e035cf...mp4 | NA | B | — |

---

## Category Summary

| Category | Answerable | Correct | Accuracy |
|----------|-----------|---------|----------|
| Action Order | 1 | 0 | 0% |
| Motion-related Objects | 2 | 2 | 100% |
| Motion Recognition | 0 | — | — |
| Location-related Motion | 1 | 0 | 0% |
| Camera Motion | 0 | — | — |
| Repetition Count | 1 | 0 | 0% |
| **Total** | **5** | **2** | **40%** |

---

## Full Run Status

| Job | ID | Node | Status |
|-----|----|------|--------|
| DyCoke | 6972461 | compute-9-1 | Running |
| Baseline | 6972462 | compute-10-7 | Running |

Results will be saved to:
- DyCoke: `/project/rhu/dpalfaro/results/dycoke_run1/`
- Baseline: `/project/rhu/dpalfaro/results/baseline_run1/`

# MotionBench Qualitative Examples — First 10 Samples

**Models:** LLaVA-OV 7B (Baseline) vs. LLaVA-OV 7B + DyCoke (l=3, p=0.8, k=0.3)
**Benchmark:** MotionBench
**Note:** Baseline predictions pending — job 6980625 in progress.
NA samples are excluded from accuracy scoring per the official MotionBench protocol.

---

## Example 1

| | |
|:---|:---|
| **Ground Truth** | C |
| **Video Type** | Gaming |
| **Question Type** | Action Order |

**Q:** Please describe the detailed breakdown of the action in the video.

| Option | Text |
|:---:|:---|
| A | Jump, Lay down, Stand |
| B | Jump, Stand, Lay down |
| **C** | **Lay down, Stand, Jump** ← correct |

| Model | Prediction | Correct? |
|:---|:---:|:---:|
| LLaVA-OV 7B (Baseline) | — | — |
| LLaVA-OV 7B + DyCoke | B | ✗ |

---

## Example 2

| | |
|:---|:---|
| **Ground Truth** | C |
| **Video Type** | Industrial |
| **Question Type** | Motion-related Objects |

**Q:** What is the boy holding in his hand?

| Option | Text |
|:---:|:---|
| A | Screw |
| B | Fountain pen |
| **C** | **Screwdriver** ← correct |
| D | Wrench |

| Model | Prediction | Correct? |
|:---|:---:|:---:|
| LLaVA-OV 7B (Baseline) | — | — |
| LLaVA-OV 7B + DyCoke | C | ✓ |

---

## Example 3 *(NA — unanswerable)*

| | |
|:---|:---|
| **Ground Truth** | NA |
| **Video Type** | Industrial |
| **Question Type** | Motion Recognition |

**Q:** What tiny movements is the person in the video doing?

| Option | Text |
|:---:|:---|
| A | Rotating object |
| B | Fist grip |
| C | Wave goodbye |
| D | Clap hands |

*This sample has no valid ground truth answer. Excluded from accuracy scoring by both models.*

---

## Example 4 *(NA — unanswerable)*

| | |
|:---|:---|
| **Ground Truth** | NA |
| **Video Type** | — |
| **Question Type** | Motion Recognition |

**Q:** Is there any interaction between multiple people in the video?

| Option | Text |
|:---:|:---|
| A | Yes, three people work together |
| B | Yes, someone hands her a box |
| C | No, she is alone |
| D | Yes, two people talk |

*This sample has no valid ground truth answer. Excluded from accuracy scoring by both models.*

---

## Example 5

| | |
|:---|:---|
| **Ground Truth** | D |
| **Video Type** | Medical |
| **Question Type** | Repetition Count |

**Q:** Please count the number of repeated actions in the video.
*(Action: Run three steps to the front right and lift your left leg, then run three steps to the front left and lift your right leg)*

| Option | Text |
|:---:|:---|
| A | 1 |
| B | 7 |
| C | 8 |
| **D** | **3** ← correct |

| Model | Prediction | Correct? |
|:---|:---:|:---:|
| LLaVA-OV 7B (Baseline) | — | — |
| LLaVA-OV 7B + DyCoke | B | ✗ |

---

## Example 6

| | |
|:---|:---|
| **Ground Truth** | C |
| **Video Type** | — |
| **Question Type** | Location-related Motion |

**Q:** What is the sequence of the power tool's movement across the nuts?

| Option | Text |
|:---:|:---|
| A | Lower-right, upper-right, upper-left, lower-left |
| B | Lower-right, upper-right, lower-left, upper-left |
| **C** | **Lower-right, upper-left, lower-left** ← correct |
| D | Lower-left, lower-right, upper-left, upper-right |

| Model | Prediction | Correct? |
|:---|:---:|:---:|
| LLaVA-OV 7B (Baseline) | — | — |
| LLaVA-OV 7B + DyCoke | A | ✗ |

---

## Example 7 *(NA — unanswerable)*

| | |
|:---|:---|
| **Ground Truth** | NA |
| **Video Type** | — |
| **Question Type** | Motion Recognition |

**Q:** Do other people's hands enter the scene?

| Option | Text |
|:---:|:---|
| A | Yes, hands appear from the right side |
| B | No, no hands enter the scene |
| C | Yes, hands appear from the left side |
| D | Yes, hands appear from above |

*This sample has no valid ground truth answer. Excluded from accuracy scoring by both models.*

---

## Example 8 *(NA — unanswerable)*

| | |
|:---|:---|
| **Ground Truth** | NA |
| **Video Type** | — |
| **Question Type** | Camera Motion |

**Q:** What occurs with the camera's focus during the woman's action?

| Option | Text |
|:---:|:---|
| A | The camera focuses on her right hand holding the gingerbread man |
| B | The camera shakes and loses focus |
| C | The camera zooms out |
| D | The camera focuses on her left hand |

*This sample has no valid ground truth answer. Excluded from accuracy scoring by both models.*

---

## Example 9

| | |
|:---|:---|
| **Ground Truth** | C |
| **Video Type** | Sports |
| **Question Type** | Motion-related Objects |

**Q:** What kind of ball did the player in the white jersey receive?

| Option | Text |
|:---:|:---|
| A | Badminton |
| B | Table tennis |
| **C** | **Baseball** ← correct |
| D | Rugby |

| Model | Prediction | Correct? |
|:---|:---:|:---:|
| LLaVA-OV 7B (Baseline) | — | — |
| LLaVA-OV 7B + DyCoke | C | ✓ |

---

## Example 10 *(NA — unanswerable)*

| | |
|:---|:---|
| **Ground Truth** | NA |
| **Video Type** | Medical |
| **Question Type** | Action Order |

**Q:** Please describe the detailed breakdown of the action in the video.

| Option | Text |
|:---:|:---|
| A | Left hand reaches over the head to touch the right ear, tilt the head to the left side |
| B | Head tilted to the left, left hand reaches over to touch the right ear |

*This sample has no valid ground truth answer. Excluded from accuracy scoring by both models.*

---

## Summary

| Sample | Category | Ground Truth | Baseline | DyCoke | Notes |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | Action Order | C | — | B ✗ | 3 options only |
| 2 | Motion-related Objects | C | — | C ✓ | |
| 3 | Motion Recognition | NA | — | — | Unanswerable |
| 4 | Motion Recognition | NA | — | — | Unanswerable |
| 5 | Repetition Count | D | — | B ✗ | |
| 6 | Location-related Motion | C | — | A ✗ | |
| 7 | Motion Recognition | NA | — | — | Unanswerable |
| 8 | Camera Motion | NA | — | — | Unanswerable |
| 9 | Motion-related Objects | C | — | C ✓ | |
| 10 | Action Order | NA | — | — | Unanswerable; 2 options only |

**DyCoke (10-sample):** 2/5 scoreable = **40%**
**Baseline:** Pending job 6980625

# iMOVE × MotionBench — NOT YET RUNNABLE

**Status: No code or weights released.**

iMOVE (Findings of ACL 2025) — "iMOVE: Instance-Motion-Aware Video Understanding"

ACL Anthology: [2025.findings-acl.1228](https://aclanthology.org/2025.findings-acl.1228/)

---

## What is iMOVE?

iMOVE is an instance-motion-aware video foundation model, designed for fine-grained instance spatiotemporal motion perception.

- **Architecture:** Trained model with instance-motion annotations + event-aware spatiotemporal modeling
- **Code status:** Authors stated code and weights will be released after acceptance — **not yet available** as of May 2026
- **No GitHub repo found**

---

## Why this folder exists

iMOVE is the most directly motion-motivated model in this benchmark set. When released, evaluating it on MotionBench would be straightforward — the benchmark directly tests instance-level motion understanding that iMOVE is designed for.

---

## What to do when code is released

1. Monitor the ACL Anthology page and author institution pages for a code release
2. Check Kuaishou Technology / Zhejiang University / CASIA GitHub orgs
3. Once released, confirm whether iMOVE provides an lmms_eval backend
4. If lmms_eval: write `run_imove.sbatch`
5. If custom: write `eval_motionbench.py` following the STTM/PruneVid pattern

---

## Do not run yet

No sbatch files or eval scripts are provided until the code is released.

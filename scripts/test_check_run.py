#!/usr/bin/env python3
"""
Tests for scripts/check_run.py — the verification gate.

Why this exists: the gate is the single piece of code every published number
depends on, and it shipped untested. A silent hole in it (`--vs-baseline`
returning early on a length mismatch) meant the divergence check was skipped on
EVERY smoke run, and two cells were briefly reported as verified when they were
actually no-ops. Tests-for-the-tester is the fix.

Run:  python3 scripts/test_check_run.py     (no deps, exits non-zero on failure)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "check_run.py")

FAILS = []


def make_run(d, preds, correct=None, params=None, n_na=0, total=None):
    """Write a minimal run dir (results.jsonl + summary.json)."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.jsonl"), "w") as f:
        for i, p in enumerate(preds):
            c = None if (correct is None and i < n_na) else (correct[i] if correct else 1)
            f.write(json.dumps({"idx": i, "video_path": f"v{i}.mp4",
                                "question_type": "Motion Recognition",
                                "ground_truth": "A", "prediction": p,
                                "correct": c}) + "\n")
    scoreable = sum(1 for i in range(len(preds)) if not (correct is None and i < n_na))
    s = {"accuracy": 0.5, "correct": scoreable // 2,
         "total_scoreable": scoreable, "total_na_skipped": n_na,
         "total_samples": total or len(preds)}
    if params:
        s.update(params)
    json.dump(s, open(os.path.join(d, "summary.json"), "w"))
    return d


def run_gate(*args):
    r = subprocess.run([sys.executable, GATE, *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


def main():
    tmp = tempfile.mkdtemp(prefix="gate_test_")
    try:
        P = {"m_params": {"enabled": True}}

        # --- the regression that motivated this suite -----------------------
        # A short smoke run vs a long baseline must STILL be compared.
        base = make_run(os.path.join(tmp, "base"), ["A"] * 100,
                        params=P, total=100)
        noop = make_run(os.path.join(tmp, "noop"), ["A"] * 8, params=P, total=8)
        rc, out = run_gate(noop, "--expect-method", "m", "--smoke",
                           "--vs-baseline", base)
        check("short smoke vs long baseline is compared, not skipped",
              "IDENTICAL to baseline" in out, "(the length-mismatch hole)")
        check("identical-to-baseline smoke FAILS", rc != 0, f"(rc={rc})")

        # --- a genuinely engaged method must pass ---------------------------
        real = make_run(os.path.join(tmp, "real"),
                        ["A", "B", "C", "D", "A", "B", "C", "D"],
                        params=P, total=8)
        rc, out = run_gate(real, "--expect-method", "m", "--smoke",
                           "--vs-baseline", base)
        check("engaged method passes", rc == 0, f"(rc={rc})")
        check("divergence is reported", "differ from baseline" in out)

        # --- empty predictions must fail ------------------------------------
        empty = make_run(os.path.join(tmp, "empty"), [""] * 8, params=P, total=8)
        rc, out = run_gate(empty, "--expect-method", "m", "--smoke")
        check("100% empty predictions FAIL", rc != 0 and "empty" in out.lower())

        # --- enabled:false must fail ----------------------------------------
        off = make_run(os.path.join(tmp, "off"), ["A", "B"] * 4,
                       params={"m_params": {"enabled": False}}, total=8)
        rc, out = run_gate(off, "--expect-method", "m", "--smoke")
        check("enabled:false FAILS", rc != 0 and "enabled" in out.lower())

        # --- missing summary must fail --------------------------------------
        broken = os.path.join(tmp, "broken")
        os.makedirs(broken, exist_ok=True)
        rc, out = run_gate(broken, "--expect-method", "m", "--smoke")
        check("missing summary.json FAILS", rc != 0)

        # --- KNOWN GAP (documented, not yet fixed): a missing baseline dir
        #     only warns. Assert current behaviour so a future change is visible.
        rc, out = run_gate(real, "--expect-method", "m", "--smoke",
                           "--vs-baseline", os.path.join(tmp, "does_not_exist"))
        check("missing baseline currently only WARNS (known gap)",
              "cannot read" in out, "(tracked in VALIDITY_ASSESSMENT)")

        # --- truncated full run must fail -----------------------------------
        trunc = make_run(os.path.join(tmp, "trunc"), ["A", "B"] * 4,
                         params=P, total=8)
        rc, out = run_gate(trunc, "--expect-method", "m")  # no --smoke
        check("truncated full run FAILS", rc != 0 and "8052" in out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print(f"{len(FAILS)} test(s) FAILED: {', '.join(FAILS)}")
        return 1
    print("all gate tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
split_by_duration.py -- does token reduction hurt more on short videos?

Motivation. FastV loses ~16 points on LLaVA-OV and the loss does not depend on
WHICH tokens survive (see docs/FASTV_COLLAPSE_ANALYSIS.md). An untested
alternative is that it depends on WHEN: at 32 uniformly sampled frames a short
clip is oversampled -- adjacent frames are near-duplicates, so a fixed token
budget may be spent very differently than on a long clip where every frame is
distinct.

This splits every cached run by source-video duration at a 5 s cut (MotionBench
median is 5.76 s, so the halves are close to balanced) and reports accuracy,
delta vs the matching backbone, and a paired McNemar test per half.

Durations come from video_info.duration in MotionBench's meta JSONL, cached to
results-cache/_video_durations.json.

    python3 analysis/duration-split/split_by_duration.py
    python3 analysis/duration-split/split_by_duration.py --cut 3 --cut 5 --cut 10
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, "results-cache")
DUR = os.path.join(CACHE, "_video_durations.json")

# (label, run dir, baseline run dir). Baseline None == this IS a baseline.

def load_run(name):
    """Return {idx: (correct, video_path, question_type)} for scoreable items."""
    p = os.path.join(CACHE, name, "results.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("ground_truth", "").strip().upper() == "NA":
            continue
        if r.get("correct") is None:
            continue
        out[r["idx"]] = (int(r["correct"]), r.get("video_path", ""), r.get("question_type", ""))
    return out


def mcnemar(a, b, keys):
    """a=reference (baseline), b=method. Returns (broke, fixed, chi2)."""
    broke = sum(1 for k in keys if a[k][0] == 1 and b[k][0] == 0)
    fixed = sum(1 for k in keys if a[k][0] == 0 and b[k][0] == 1)
    n = broke + fixed
    if n == 0:
        return broke, fixed, 0.0
    chi2 = (abs(broke - fixed) - 1) ** 2 / n
    return broke, fixed, chi2


def interaction(base, run, short, long_):
    """Is the damage profile different on short vs long videos?

    2x2 chi-square of independence on [half] x [broke, fixed]. A method whose
    loss is duration-driven should break-vs-fix at different rates in the two
    halves; one whose loss is duration-invariant should not.
    """
    bs, fs, _ = mcnemar(base, run, short)
    bl, fl, _ = mcnemar(base, run, long_)
    n = bs + fs + bl + fl
    if n == 0 or (bs + fs) == 0 or (bl + fl) == 0:
        return 0.0, False
    rb, rf = bs + bl, fs + fl
    rs, rl = bs + fs, bl + fl
    chi2 = 0.0
    for obs, (r, c) in zip((bs, fs, bl, fl), ((rs, rb), (rs, rf), (rl, rb), (rl, rf))):
        exp = r * c / n
        if exp > 0:
            chi2 += (abs(obs - exp) - 0.5) ** 2 / exp
    return chi2, chi2 >= 3.84


def acc(run, keys):
    if not keys:
        return float("nan"), 0, 0
    c = sum(run[k][0] for k in keys)
    return 100.0 * c / len(keys), c, len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=float, action="append", default=None,
                    help="duration cut in seconds (repeatable); default 5")
    ap.add_argument("--frames", type=int, default=32,
                    help="frames sampled per video, for the oversampling column")
    args = ap.parse_args()
    cuts = args.cut or [5.0]

    if not os.path.exists(DUR):
        sys.exit("missing %s -- see module docstring" % DUR)
    dur = {k: v[0] for k, v in json.load(open(DUR)).items()}

    # fastv_run1 is the original no-op FastV port: apply_fastv() was a stub, so
    # it is bit-identical to the plain backbone (52.66%) and is what every
    # LLaVA-OV sbatch passes to check_run.py as --vs-baseline.
    LLAVA_BASE = "fastv_run1"
    QWEN_BASE = "qwen3vl_baseline_run1"

    groups = [
        ("LLaVA-OV-7B, keep 15%", LLAVA_BASE, [
            ("FastV", "w2_fastv_run"),
            ("FlashVID", "w2_flashvid_run"),
            ("PruneVID-OV", "w2_prunevid_ov_run"),
            ("DyCoke", "w2_dycoke_run"),
            ("HoliTom", "w2_holitom_run"),
            ("AIM", "w2_aim_run"),
            ("MDP3", "w2_mdp3_run"),
            ("VideoITG", "w2_videoitg_run"),
            ("STTM", "w2_sttm_run"),
        ]),
        ("LLaVA-OV-7B, keep 10%", LLAVA_BASE, [
            ("FastV", "s1_fastv_r10_run"),
            ("FlashVID", "s1_flashvid_r10_run"),
            ("PruneVID-OV", "s1_prunevid_r10_run"),
            ("HoliTom", "s1_holitom_r10_run"),
        ]),
        ("LLaVA-OV-7B, keep 25%", LLAVA_BASE, [
            ("FastV", "s1_fastv_r25_run"),
            ("FlashVID", "s1_flashvid_r25_run"),
            ("PruneVID-OV", "s1_prunevid_r25_run"),
            ("HoliTom", "s1_holitom_r25_run"),
        ]),
        ("Qwen3-VL-8B, keep 15%", QWEN_BASE, [
            ("FastV", "w3_fastv_run"),
            ("FlashVID", "w3_flashvid_run"),
            ("PruneVID", "w3_prunevid_run"),
            ("DyCoke", "w3_dycoke_run"),
            ("HoliTom", "w3_holitom_run"),
            ("AIM", "w3_aim_run"),
        ]),
    ]

    for cut in cuts:
        print("\n" + "=" * 100)
        print("DURATION SPLIT AT %.1f s   (frames=%d -> a %.1fs clip is sampled at %.1f fps)"
              % (cut, args.frames, cut, args.frames / cut))
        print("=" * 100)

        for title, base_name, methods in groups:
            base = load_run(base_name)
            if base is None:
                print("\n[%s] baseline %s not cached -- skipped" % (title, base_name))
                continue

            # Partition baseline keys by duration.
            short = [k for k, v in base.items() if dur.get(v[1]) is not None and dur[v[1]] < cut]
            long_ = [k for k, v in base.items() if dur.get(v[1]) is not None and dur[v[1]] >= cut]
            unk = len(base) - len(short) - len(long_)

            bs, _, ns = acc(base, short)
            bl, _, nl = acc(base, long_)
            print("\n%s   [short n=%d  long n=%d  unmatched=%d]" % (title, ns, nl, unk))
            print("  baseline: short %.2f   long %.2f   (gap %+.2f)" % (bs, bl, bl - bs))
            print("  %-14s %18s %18s %10s %14s" % ("method", "SHORT <%.0fs" % cut,
                  "LONG >=%.0fs" % cut, "asym", "interaction"))
            print("  %-14s %18s %18s %10s %14s" % ("", "acc    d   chi2", "acc    d   chi2",
                  "dS-dL", "chi2  signif"))

            for label, rd in methods:
                run = load_run(rd)
                if run is None:
                    print("  %-14s %s" % (label, "(not cached)"))
                    continue
                ks = [k for k in short if k in run]
                kl = [k for k in long_ if k in run]
                as_, _, _ = acc(run, ks)
                al_, _, _ = acc(run, kl)
                b0s, _, _ = acc(base, ks)
                b0l, _, _ = acc(base, kl)
                _, _, cs = mcnemar(base, run, ks)
                _, _, cl = mcnemar(base, run, kl)
                ds, dl = as_ - b0s, al_ - b0l
                ix, sig = interaction(base, run, ks, kl)
                print("  %-14s  %6.2f %+6.2f %6.1f  %6.2f %+6.2f %6.1f   %+6.2f   %6.2f  %s"
                      % (label, as_, ds, cs, al_, dl, cl, ds - dl, ix,
                         "YES" if sig else "no"))


if __name__ == "__main__":
    main()

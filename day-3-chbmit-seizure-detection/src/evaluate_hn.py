"""Improvement-pass evaluation. Compares, on the SAME held-out patients:
  v1          CNN as first evaluated (5 s smoothing)
  v1 + tuned  same CNN, smoothing window chosen on validation patients
  hn + tuned  hard-negative CNN, smoothing window chosen on validation patients
The operating point (smoothing, threshold) is always chosen on validation patients only.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import evaluate as ev
from data import ROOT, TEST, VAL, window_end_times

R = ROOT / "results"
KS = (5, 15, 30, 60)

# Pre-registered before the run (see README): both must hold on the test patients.
CRITERIA = {"max_fa_per_hour": 2.0, "min_sensitivity": 0.40}


def choose(val, key):
    best = None
    for k in KS:
        sw = ev.sweep(val, key, k)
        ok = [(t, r) for t, r in sw if r["fa_per_hour"] <= ev.FA_BUDGET]
        if not ok:
            continue
        t, r = max(ok, key=lambda x: (x[1]["sensitivity"], -x[1]["fa_per_hour"]))
        cand = (r["sensitivity"], -r["fa_per_hour"], k, t)
        if best is None or cand > best:
            best = cand
    if best is None:  # nothing met the budget on validation: fall back to the strictest operating point
        return KS[-1], ev.THRS[-1]
    return best[2], best[3]


def summarise(res):
    return {k: v for k, v in res.items() if k != "per_case"}


def per_case(res):
    return {c: {"detected": v[0], "seizures": v[1], "false_alarms": v[2], "fa_per_24h": round(v[2] / v[3] * 24, 2)}
            for c, v in res["per_case"].items()}


def main():
    v1_val, v1_test = ev.load(VAL), ev.load(TEST)
    hn_val, hn_test = ev.load(VAL, "scores_hn"), ev.load(TEST, "scores_hn")
    out = {"criteria": CRITERIA, "settings": {"fa_budget_per_h": ev.FA_BUDGET, "smoothing_grid_s": list(KS)}}
    curves = {}
    runs = {
        "v1 (5 s smoothing)": (v1_test, "cnn", 5, json.load(open(R / "events.json"))["CNN"]["threshold_from_val"], None),
        "v1 + tuned smoothing": (v1_test, "cnn", *choose(v1_val, "cnn")[::1], v1_val),
        "hard-negative + tuned smoothing": (hn_test, "cnn", *choose(hn_val, "cnn")[::1], hn_val),
    }
    for name, (test, key, k, thr, val) in runs.items():
        at = ev.run(test, key, thr, k)
        sv = ev.run(val, key, thr, k) if val is not None else None
        curves[name] = [(r["fa_per_hour"], r["sensitivity"]) for _, r in ev.sweep(test, key, k)]
        out[name] = {"smoothing_s": k, "threshold_from_val": float(thr), "val": summarise(sv) if sv else None,
                     "test": summarise(at), "test_per_case": per_case(at)}
        print(f"{name}: smoothing {k}s thr {thr:.3f} | test {at['detected']}/{at['seizures']} seizures, "
              f"{at['fa_per_hour']:.2f} FA/h, median latency {at['median_latency_s']}")
    final = out["hard-negative + tuned smoothing"]["test"]
    out["criteria_met"] = bool(final["fa_per_hour"] <= CRITERIA["max_fa_per_hour"] and final["sensitivity"] >= CRITERIA["min_sensitivity"])
    print("CRITERIA MET:", out["criteria_met"], CRITERIA)
    (R / "events_hn.json").write_text(json.dumps(out, indent=2))

    plt.figure(figsize=(6.2, 4.8))
    for name, pts in curves.items():
        pts = sorted(pts)
        plt.step([a for a, _ in pts], [b for _, b in pts], where="post", label=name)
    plt.xscale("symlog", linthresh=0.05); plt.xlim(0, 30); plt.ylim(0, 1.02)
    plt.axvline(CRITERIA["max_fa_per_hour"], color="gray", lw=0.8, ls="--")
    plt.xlabel("False alarms per hour (held-out patients)"); plt.ylabel("Seizure sensitivity")
    plt.title("Improvement pass: held-out patients"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(R / "sens_vs_fa_hn.png", dpi=150); plt.close()

    # same recording, before and after (the chb05 file with the worst post-seizure false alarms)
    if not (R / "timeline.json").exists():
        return  # timeline.json comes from evaluate.py; skip the before/after figure without it
    tl = json.load(open(R / "timeline.json"))
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, (name, recs, key, k, thr) in zip(axs, [("v1", v1_test, "cnn", 5, runs["v1 (5 s smoothing)"][3]),
                                                  ("hard-negative", hn_test, "cnn", *runs["hard-negative + tuned smoothing"][2:4])]):
        r = next(x for x in recs if x["case"] == tl["case"] and x["file"] == tl["file"])
        s = ev.smooth(r[key], k); t = window_end_times(len(s)); on, off = tl["onset"], tl["offset"]
        m = (t >= max(on - 600, 0)) & (t <= min(off + 600, r["dur"]))
        ax.plot((t[m] - on) / 60, s[m], color="#ff2d55", lw=1.6)
        ax.axvspan(0, (off - on) / 60, color="#ff2d55", alpha=0.12)
        ax.axhline(thr, color="k", ls="--", lw=0.8)
        ax.set_ylim(0, 1.02); ax.set_ylabel("P(seizure)"); ax.set_title(f"{name}: {tl['case']} {tl['file']}", fontsize=10)
    axs[-1].set_xlabel("minutes from annotated seizure onset")
    fig.tight_layout(); fig.savefig(R / "timeline_hn.png", dpi=150)


if __name__ == "__main__":
    main()

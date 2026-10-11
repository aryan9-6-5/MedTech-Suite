"""Out-of-fold event evaluation over the 19 development patients. Decides whether the dev criterion is met."""
import json

import matplotlib.pyplot as plt
import numpy as np

import evaluate as ev
from data import DEV, ROOT

R = ROOT / "results"
KS = (5, 15, 30, 60)
CRIT = {"max_fa_per_hour": 2.0, "min_sensitivity": 0.50}  # pre-registered, see README


def choose(recs):
    best = None
    for k in KS:
        for thr, r in ev.sweep(recs, "cnn", k):
            if r["fa_per_hour"] <= CRIT["max_fa_per_hour"]:
                cand = (r["sensitivity"], -r["fa_per_hour"], k, thr)
                if best is None or cand > best:
                    best = cand
    return (best[2], best[3]) if best else (KS[-1], ev.THRS[-1])


def main():
    out, curves = {"criteria": CRIT}, {}
    for v, label in (("z", "z-score only"), ("rel", "z-score + baseline-relative")):
        recs = ev.load(DEV, f"cv/{v}")
        k, thr = choose(recs)
        at = ev.run(recs, "cnn", thr, k)
        curves[label] = sorted((r["fa_per_hour"], r["sensitivity"]) for _, r in ev.sweep(recs, "cnn", k))
        out[v] = {"smoothing_s": k, "threshold": float(thr), "files": len(recs),
                  **{a: b for a, b in at.items() if a != "per_case"},
                  "per_patient": {c: {"detected": p[0], "seizures": p[1], "false_alarms": p[2], "fa_per_24h": round(p[2] / p[3] * 24, 2)}
                                  for c, p in at["per_case"].items()}}
        print(f"{label}: smoothing {k}s | {at['detected']}/{at['seizures']} seizures ({at['sensitivity']:.1%}), "
              f"{at['fa_per_hour']:.2f} FA/h, median latency {at['median_latency_s']}")
    best = max(("z", "rel"), key=lambda v: (out[v]["sensitivity"], -out[v]["fa_per_hour"]))
    ok = out[best]["fa_per_hour"] <= CRIT["max_fa_per_hour"] and out[best]["sensitivity"] >= CRIT["min_sensitivity"]
    out.update(best_variant=best, criteria_met=bool(ok))
    print("DEV CRITERIA MET:", ok, "| best variant:", best, CRIT)
    (R / "cv").mkdir(exist_ok=True)
    (R / "cv" / "summary.json").write_text(json.dumps(out, indent=2))

    plt.figure(figsize=(6.2, 4.8))
    for label, pts in curves.items():
        plt.step([a for a, _ in pts], [s for _, s in pts], where="post", label=label)
    plt.xscale("symlog", linthresh=0.05); plt.xlim(0, 30); plt.ylim(0, 1.02)
    plt.axvline(CRIT["max_fa_per_hour"], color="gray", lw=0.8, ls="--"); plt.axhline(CRIT["min_sensitivity"], color="gray", lw=0.8, ls=":")
    plt.xlabel("False alarms per hour (19 development patients, out-of-fold)"); plt.ylabel("Seizure sensitivity")
    plt.title("Cross-validated, every patient held out once"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(R / "cv" / "sens_vs_fa_cv.png", dpi=150)


if __name__ == "__main__":
    main()

"""Event-level evaluation: seizures detected, false alarms per hour, latency. Threshold chosen on validation patients."""
import json

import matplotlib.pyplot as plt
import numpy as np

from data import ROOT, TEST, VAL, window_end_times

R = ROOT / "results"
SMOOTH, REFRACTORY, GRACE, FA_BUDGET = 5, 300, 30, 1.0  # seconds, seconds, seconds, alarms per hour
THRS = 1 / (1 + np.exp(-np.linspace(-4, 9, 120)))


def load(cases):
    out = []
    for f in sorted((R / "scores").glob("*.npz")):
        if f.name.split("__")[0] in cases:
            z = np.load(f)
            out.append({"case": f.name.split("__")[0], "file": f.name.split("__")[1][:-4], "cnn": z["cnn"].astype(np.float32),
                        "base": z["base"].astype(np.float32), "dur": float(z["duration"]), "sz": z["seizures"]})
    return out


def smooth(p, k=SMOOTH):
    c = np.cumsum(np.insert(p, 0, 0.0))  # trailing mean: causal
    s = np.empty_like(p)
    for i in range(len(p)):
        lo = max(0, i - k + 1)
        s[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return s


def alarms(s, t, thr):
    out, last = [], -1e9
    for i in np.flatnonzero(s >= thr):
        if t[i] - last >= REFRACTORY:
            out.append(t[i]); last = t[i]
    return np.array(out)


def run(recs, key, thr):
    """Totals over recordings at one threshold. Alarm = smoothed probability crosses thr (5 min refractory)."""
    det = nsz = fa = 0
    hours, lat, per = 0.0, [], {}
    for r in recs:
        s = smooth(r[key]); t = window_end_times(len(s))
        al = alarms(s, t, thr)
        sz = r["sz"]
        excl = sum(min(off + GRACE, r["dur"]) - on for on, off in sz)
        h = (r["dur"] - excl) / 3600
        d = 0
        for on, off in sz:
            hit = al[(al >= on) & (al <= off + GRACE)]
            if len(hit):
                d += 1; lat.append(float(hit[0] - on))
        f = sum(not any(on <= a <= off + GRACE for on, off in sz) for a in al)
        det += d; nsz += len(sz); fa += f; hours += h
        p = per.setdefault(r["case"], [0, 0, 0, 0.0]); p[0] += d; p[1] += len(sz); p[2] += f; p[3] += h
    return {"detected": det, "seizures": nsz, "sensitivity": det / max(nsz, 1), "false_alarms": fa, "hours": hours,
            "fa_per_hour": fa / max(hours, 1e-9), "median_latency_s": float(np.median(lat)) if lat else None, "per_case": per}


def sweep(recs, key):
    return [(thr, run(recs, key, thr)) for thr in THRS]


def pick(sw):
    ok = [(t, r) for t, r in sw if r["fa_per_hour"] <= FA_BUDGET]
    if not ok:
        return sw[-1][0]
    return max(ok, key=lambda x: (x[1]["sensitivity"], -x[1]["fa_per_hour"]))[0]


def main():
    val, test = load(VAL), load(TEST)
    print(f"val files {len(val)}, test files {len(test)}")
    out = {"settings": {"smoothing_s": SMOOTH, "refractory_s": REFRACTORY, "grace_s": GRACE, "fa_budget_per_h": FA_BUDGET}}
    curves = {}
    for key, name in (("cnn", "CNN"), ("base", "Baseline")):
        thr = pick(sweep(val, key))
        sw_test = sweep(test, key)
        at = run(test, key, thr)
        curves[name] = [(r["fa_per_hour"], r["sensitivity"]) for _, r in sw_test]
        out[name] = {"threshold_from_val": float(thr), "test": {k: v for k, v in at.items() if k != "per_case"},
                     "test_per_case": {c: {"detected": v[0], "seizures": v[1], "false_alarms": v[2], "hours": round(v[3], 1),
                                           "fa_per_24h": round(v[2] / v[3] * 24, 2) if v[3] else None} for c, v in at["per_case"].items()}}
        print(name, json.dumps(out[name]["test"], indent=1))
    (R / "events.json").write_text(json.dumps(out, indent=2))

    plt.figure(figsize=(6.2, 4.8))
    for name, pts in curves.items():
        pts = sorted(pts)
        plt.step([a for a, _ in pts], [b for _, b in pts], where="post", label=name)
    plt.xscale("symlog", linthresh=0.05); plt.xlim(0, 30); plt.ylim(0, 1.02)
    plt.axvline(FA_BUDGET, color="gray", lw=0.8, ls="--")
    plt.xlabel("False alarms per hour (held-out patients)"); plt.ylabel("Seizure sensitivity")
    plt.title("Held-out patients: sensitivity vs false alarms"); plt.legend(); plt.tight_layout()
    plt.savefig(R / "sens_vs_fa.png", dpi=150); plt.close()

    # timeline for the held-out recording with the clearest detection
    thr = out["CNN"]["threshold_from_val"]
    best = None
    for r in test:
        if len(r["sz"]):
            s = smooth(r["cnn"]); on, off = r["sz"][0]
            t = window_end_times(len(s))
            m = (t >= on) & (t <= off)
            sc = s[m].max() if m.any() else 0
            if best is None or sc > best[0]:
                best = (sc, r, s, t, on, off)
    _, r, s, t, on, off = best
    lo, hi = max(on - 600, 0), min(off + 600, r["dur"])
    m = (t >= lo) & (t <= hi)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot((t[m] - on) / 60, r["cnn"][m], color="#c9c9ce", lw=0.8, label="CNN output per second")
    ax.plot((t[m] - on) / 60, s[m], color="#ff2d55", lw=1.8, label="smoothed (5 s)")
    ax.axvspan(0, (off - on) / 60, color="#ff2d55", alpha=0.12, label="annotated seizure")
    ax.axhline(thr, color="k", ls="--", lw=0.8, label="alarm threshold (set on validation patients)")
    ax.set_xlabel("minutes from annotated seizure onset"); ax.set_ylabel("P(seizure)"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Held-out patient {r['case']}, {r['file']}"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(R / "timeline.png", dpi=150)
    (R / "timeline.json").write_text(json.dumps({"case": r["case"], "file": r["file"], "onset": float(on), "offset": float(off)}))


if __name__ == "__main__":
    main()

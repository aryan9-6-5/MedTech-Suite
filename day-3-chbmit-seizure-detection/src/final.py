"""One-shot test evaluation. Refuses to run unless the development (cross-validated) criterion was met.

Trains on all 19 development patients with the variant, smoothing and threshold chosen in cv_eval.py, then scores
the 5 test patients once. (These patients were seen in experiments 1 and 2; see README for that caveat.)
"""
import json
import sys

import numpy as np

import evaluate as ev
from cache2 import load2
from data import DATA, DEV, ROOT, TEST, parse_summary, read_edf
from fit import fit, score_signal

R = ROOT / "results"


def main(force=False):
    s = json.load(open(R / "cv" / "summary.json"))
    if not s["criteria_met"] and not force:
        sys.exit("Development criterion not met; not touching the test patients. (--force to override, and say so in the write-up.)")
    v = s["best_variant"]; k = s[v]["smoothing_s"]; thr = s[v]["threshold"]
    X, y, b, _ = load2(DEV)
    model = fit(X, y, b, np.arange(len(y)), v == "rel")
    out = R / "final"; out.mkdir(exist_ok=True)
    for case in TEST:
        for fname, szs in parse_summary(case).items():
            p = DATA / case / fname
            x = read_edf(p) if p.exists() else None
            if x is None:
                continue
            np.savez_compressed(out / f"{case}__{fname}.npz", cnn=score_signal(model, v == "rel", x).astype(np.float16),
                                duration=x.shape[1] / 256, seizures=np.array(szs, dtype=float).reshape(-1, 2))
            print(case, fname, flush=True)
    at = ev.run(ev.load(TEST, "final"), "cnn", thr, k)
    res = {"variant": v, "smoothing_s": k, "threshold": thr, **{a: c for a, c in at.items() if a != "per_case"},
           "per_patient": {c: {"detected": p[0], "seizures": p[1], "false_alarms": p[2], "fa_per_24h": round(p[2] / p[3] * 24, 2)}
                           for c, p in at["per_case"].items()},
           "test_criteria_met": bool(at["fa_per_hour"] <= 2.0 and at["detected"] >= 14), "forced": force}
    (R / "events_final.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({a: c for a, c in res.items() if a != "per_patient"}, indent=2))


if __name__ == "__main__":
    main("--force" in sys.argv)

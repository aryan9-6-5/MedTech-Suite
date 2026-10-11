"""Experiment 2: patient-grouped 5-fold cross-validation over the 19 development patients.

Every development recording is scored by a model that never saw that patient. Two input variants are compared:
'z' (per-window z-score, as before) and 'rel' (z-score plus the window relative to the patient's own baseline).
The 5 test patients are not touched here.
"""
import numpy as np
import torch

from cache2 import load2
from data import DATA, DEV, ROOT, CASES, dev_folds, parse_summary, read_edf
from fit import fit, score_signal

CV = ROOT / "results" / "cv"


def run(folds=None, epochs=12, variants=("z", "rel")):
    folds = folds or dev_folds()
    X, y, b, c = load2(DEV)
    names = np.array(CASES)[c]
    for f, held in enumerate(folds):
        done = CV / f"fold{f}.done"
        if done.exists():
            continue
        print(f"fold {f}: held out {held}", flush=True)
        tr = np.flatnonzero(~np.isin(names, held))
        models = {v: fit(X, y, b, tr, v == "rel", epochs) for v in variants}
        for v, m in models.items():
            (CV / v).mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), CV / f"fold{f}_{v}.pt")
        for case in held:
            for fname, szs in parse_summary(case).items():
                p = DATA / case / fname
                x = read_edf(p) if p.exists() else None
                if x is None:
                    continue
                for v, m in models.items():
                    np.savez_compressed(CV / v / f"{case}__{fname}.npz",
                                        cnn=score_signal(m, v == "rel", x).astype(np.float16),
                                        duration=x.shape[1] / 256, seizures=np.array(szs, dtype=float).reshape(-1, 2))
                print(case, fname, flush=True)
        done.write_text("ok")


if __name__ == "__main__":
    run()

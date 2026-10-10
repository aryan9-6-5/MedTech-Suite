"""Band-power features + gradient boosting. SPLIT=patient (default) or SPLIT=random (leaky comparison)."""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from cache import load_cache, make_split
from data import ROOT
from features import features

MODE = os.environ.get("SPLIT", "patient")


def feats(X, idx):
    return np.concatenate([features(X[idx[i:i + 2048]]) for i in range(0, len(idx), 2048)])


def wl(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "prevalence": float(y.mean()), "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    res = ROOT / "results"; res.mkdir(exist_ok=True)
    X, y, c = load_cache()
    tr, va, te = (np.flatnonzero(m) for m in make_split(c, MODE))
    Ftr, Fva, Fte = feats(X, tr), feats(X, va), feats(X, te)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=42)
    clf.fit(Ftr, y[tr])
    out = {"mode": MODE, "model": "baseline", "val": wl(y[va], clf.predict_proba(Fva)[:, 1]),
           "test": wl(y[te], clf.predict_proba(Fte)[:, 1])}
    joblib.dump(clf, res / f"baseline_{MODE}.joblib")
    (res / f"windows_baseline_{MODE}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

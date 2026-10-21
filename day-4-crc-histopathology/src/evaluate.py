"""Internal vs external metrics for both arms and both seeds, and the pre-registered decision."""
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from data import CLASSES, ROOT

R = ROOT / "results"
SEEDS = (42, 7)
MIN_GAIN = 0.01  # pre-registered: arm B must beat arm A by at least 1 point of external macro-F1, in the mean and in both seeds


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def metrics(p, y):
    pred = p.argmax(1)
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "ece": ece(p, y),
            "recall": dict(zip(CLASSES, recall_score(y, pred, average=None, labels=range(9)).round(4).tolist()))}


def boot_ci(p, y, n=500, seed=0):
    """Patch-level bootstrap of external macro-F1. Optimistic: patches from one patient are correlated and patient IDs are not released."""
    rng = np.random.default_rng(seed)
    pred = p.argmax(1)
    v = [f1_score(y[i], pred[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    out, runs = {"min_gain_macro_f1": MIN_GAIN}, {}
    for arm in ("A", "B"):
        for s in SEEDS:
            z = np.load(R / f"preds_{arm}_s{s}.npz")
            runs[(arm, s)] = {"internal_test": metrics(z["test"], z["test_y"]), "external": metrics(z["ext"], z["ext_y"]),
                              "external_macro_f1_ci95_patch_bootstrap": boot_ci(z["ext"], z["ext_y"])}
    out["runs"] = {f"{a}_seed{s}": v for (a, s), v in runs.items()}
    for arm in ("A", "B"):
        out[f"{arm}_mean"] = {k: float(np.mean([runs[(arm, s)][k][m] for s in SEEDS]))
                              for k, m in (("internal_test", "macro_f1"), ("external", "macro_f1"))} | {
            "external_accuracy": float(np.mean([runs[(arm, s)]["external"]["accuracy"] for s in SEEDS])),
            "internal_accuracy": float(np.mean([runs[(arm, s)]["internal_test"]["accuracy"] for s in SEEDS]))}
    gains = [runs[("B", s)]["external"]["macro_f1"] - runs[("A", s)]["external"]["macro_f1"] for s in SEEDS]
    out["B_minus_A_external_macro_f1_by_seed"] = gains
    out["criterion_met"] = bool(np.mean(gains) >= MIN_GAIN and all(g > 0 for g in gains))
    (R / "summary.json").write_text(json.dumps(out, indent=2))
    for arm in ("A", "B"):
        print(arm, "internal macro-F1 %.4f | external macro-F1 %.4f acc %.4f" % (
            out[f"{arm}_mean"]["internal_test"], out[f"{arm}_mean"]["external"], out[f"{arm}_mean"]["external_accuracy"]))
    print("B - A external macro-F1 by seed:", [round(g, 4) for g in gains], "| criterion met:", out["criterion_met"])

    z = np.load(R / "preds_A_s42.npz")
    cm = np.nan_to_num(confusion_matrix(z["ext_y"], z["ext"].argmax(1), labels=list(range(9)), normalize="true"))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASSES, rotation=45); ax.set_yticks(range(9), CLASSES)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true (row-normalised = recall)"); ax.set_title("External set (CRC-VAL-HE-7K), arm A seed 42")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(R / "confusion_external.png", dpi=150)


if __name__ == "__main__":
    main()

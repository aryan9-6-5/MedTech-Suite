"""Qualitative figure: six random ETIS-Larib images (fixed draw) with ground truth (green) and each encoder's seed-42 prediction (red)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import ROOT, load

d = load()
X, M = d["X_etis"], d["M_etis"]
idx = np.sort(np.random.default_rng(0).choice(len(X), 6, replace=False))
preds = {e: np.load(ROOT / "results" / f"preds_{e}_s42.npz")["etis"] for e in ("resnet34", "convnext")}
fig, ax = plt.subplots(3, 6, figsize=(15, 7.8))
for j, i in enumerate(idx):
    for r, (title, p) in enumerate([("image + ground truth", None), ("ResNet-34", preds["resnet34"]), ("ConvNeXt-Tiny", preds["convnext"])]):
        a = ax[r, j]
        a.imshow(X[i])
        a.contour(M[i], levels=[0.5], colors="lime", linewidths=1.5)
        if p is not None:
            a.contour(p[i] > 127, levels=[0.5], colors="red", linewidths=1.5)
        a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a.set_ylabel(title)
fig.suptitle("ETIS-Larib (external, unseen during training): ground truth green, prediction red", y=0.94)
fig.tight_layout()
fig.savefig(ROOT / "results" / "overlays_etis.png", dpi=110)

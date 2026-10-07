"""ROC curves, F1 at 0.5, and a gradient-saliency demo figure on a test ECG."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_curve

from data import CLASSES, ROOT, load_meta, load_signals, splits, standardize
from model import ECGResNet

R = ROOT / "results"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    df, y = load_meta()
    raw = load_signals(df)
    tr, _, _ = splits(df)
    x = standardize(raw, tr)
    idx = np.load(R / "test_idx.npy")
    p = np.load(R / "test_probs.npy")
    yt = y[idx]

    plt.figure(figsize=(6, 5))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yt[:, i], p[:, i])
        plt.plot(fpr, tpr, label=c)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("PTB-XL test fold 10: ROC per superclass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(R / "roc.png", dpi=150)
    plt.close()

    f1 = {c: float(f1_score(yt[:, i], p[:, i] > 0.5)) for i, c in enumerate(CLASSES)}
    (R / "f1_at_0.5.json").write_text(json.dumps(f1, indent=2))

    # Saliency demo: most confident single-label MI record in the test fold
    mi = CLASSES.index("MI")
    cand = np.where((yt[:, mi] == 1) & (yt.sum(1) == 1))[0]
    k = cand[np.argmax(p[cand, mi])]
    sig = torch.from_numpy(x[idx[k]]).T.unsqueeze(0).float().requires_grad_(True)
    model = ECGResNet()
    model.load_state_dict(torch.load(R / "best.pt", map_location="cpu"))
    model.eval()
    model(sig)[0, mi].backward()
    sal = sig.grad.abs()[0].numpy().mean(0)
    sal = np.convolve(sal, np.ones(25) / 25, mode="same")
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

    fig, axs = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    t = np.arange(1000) / 100
    for ax, lead in zip(axs, [1, 6, 7, 8]):
        s = raw[idx[k]][:, lead]
        ax.plot(t, s, "k", lw=0.9)
        ax.imshow(sal[None], aspect="auto", cmap="Reds", alpha=0.45,
                  extent=[0, 10, s.min(), s.max()], zorder=0)
        ax.set_ylabel(LEADS[lead])
        ax.set_xlim(0, 10)
    axs[-1].set_xlabel("seconds")
    fig.suptitle(f"Test ECG {df.index[idx[k]]}: true MI, predicted P(MI)={p[k, mi]:.2f}\n"
                 "red = where the MI logit is most sensitive to the input")
    fig.tight_layout()
    fig.savefig(R / "saliency_mi.png", dpi=150)
    print("F1@0.5", f1)


if __name__ == "__main__":
    main()

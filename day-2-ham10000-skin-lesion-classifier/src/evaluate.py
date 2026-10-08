"""Confusion matrix, ROC curves, and Grad-CAM on test images (confident hits and misses)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve

from data import CLASSES, NAMES, ROOT, load_images, load_meta
from train import build_model, dev, prep

R = ROOT / "results"


def gradcam(model, x, cls):
    feats = {}
    layer = model.features[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: feats.update(a=o))
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x)
    h1.remove()
    g = torch.autograd.grad(out[0, cls], feats["a"])[0]
    cam = torch.relu((g.mean((2, 3), keepdim=True) * feats["a"]).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-9)).detach().cpu().numpy()


def main():
    df = load_meta()
    X = load_images(df)
    te = np.load(R / "test_idx.npy")
    p = np.load(R / "test_probs.npy")
    yt = df.y.values[te]
    pred = p.argmax(1)

    cm = confusion_matrix(yt, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), CLASSES, rotation=45)
    ax.set_yticks(range(7), CLASSES)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalised = recall)")
    ax.set_title("HAM10000 test split: confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(R / "confusion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yt == i, p[:, i])
        plt.plot(fpr, tpr, label=c)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.title("HAM10000 test split: ROC per class")
    plt.tight_layout()
    plt.savefig(R / "roc.png", dpi=150)
    plt.close()

    model = build_model().to(dev)
    model.load_state_dict(torch.load(R / "best.pt", map_location=dev))
    mel = CLASSES.index("mel")
    # most confident correct melanoma, and the most confident melanoma the model missed
    hit = np.where((yt == mel) & (pred == mel))[0]
    miss = np.where((yt == mel) & (pred != mel))[0]
    picks = [("correct", hit[np.argmax(p[hit, mel])])]
    if len(miss):
        picks.append(("missed", miss[np.argmax(p[miss].max(1))]))
    fig, axs = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 6.2), squeeze=False)
    for col, (tag, k) in enumerate(picks):
        img = X[te[k]]
        xb = prep(torch.from_numpy(img[None]), False)
        cam = gradcam(model, xb, mel)
        axs[0, col].imshow(img)
        axs[0, col].set_title(f"{tag}: true {NAMES['mel']}\npredicted {NAMES[CLASSES[pred[k]]]} ({p[k].max():.2f})", fontsize=9)
        axs[1, col].imshow(img)
        axs[1, col].imshow(cam, cmap="jet", alpha=0.4)
        axs[1, col].set_title("Grad-CAM for the melanoma output", fontsize=9)
        for a in axs[:, col]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(R / "gradcam_mel.png", dpi=150)
    (R / "gradcam_picks.json").write_text(json.dumps(
        [{"tag": t, "image_id": df.image_id.iloc[te[k]], "pred": CLASSES[pred[k]], "conf": float(p[k].max())} for t, k in picks], indent=2))
    print("done", picks)


if __name__ == "__main__":
    main()

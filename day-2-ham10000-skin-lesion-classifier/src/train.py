import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data import CLASSES, ROOT, load_images, load_meta, make_splits

SEED, EPOCHS, BS, LR = 42, 15, 64, 3e-4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def build_model():
    m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(CLASSES))
    return m


def prep(xb, train):
    """uint8 NHWC -> normalised NCHW float; light per-sample augmentation on the GPU."""
    x = xb.to(dev).permute(0, 3, 1, 2).float() / 255
    if train:
        n = x.size(0)
        flip_h = torch.rand(n, device=dev) < 0.5
        flip_v = torch.rand(n, device=dev) < 0.5
        x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(3), x)
        x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(2), x)
        x = x * (1 + 0.2 * (torch.rand(n, 1, 1, 1, device=dev) - 0.5))  # brightness
        mu = x.mean((1, 2, 3), keepdim=True)
        x = (x - mu) * (1 + 0.2 * (torch.rand(n, 1, 1, 1, device=dev) - 0.5)) + mu  # contrast
        x = x.clamp(0, 1)
    return (x - MEAN) / STD


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for i in range(0, len(X), 128):
            out.append(F.softmax(model(prep(X[i:i + 128], False)).float(), 1).cpu())
    return torch.cat(out).numpy()


def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in range(len(CLASSES))]))


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    df = load_meta()
    split = make_splits(df)
    X = torch.from_numpy(load_images(df))
    y = df.y.values
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    print({s: int((split == s).sum()) for s in ("train", "val", "test")}, flush=True)
    assert not set(df.lesion_id[tr]) & set(df.lesion_id[te]) and not set(df.lesion_id[tr]) & set(df.lesion_id[va])

    model = build_model().to(dev)
    freq = np.bincount(y[tr], minlength=len(CLASSES)) / len(tr)
    w = torch.tensor((1 / freq) ** 0.5, dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w / w.mean(), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=0.05)
    steps = EPOCHS * (len(tr) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    best, hist = 0.0, []
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            xb, yb = prep(X[idx], True), torch.from_numpy(y[idx]).to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        pv = predict(model, X[va])
        auc = macro_auc(y[va], pv)  # model selection on the validation split only
        hist.append({"epoch": ep + 1, "val_macro_auc": auc,
                     "val_balanced_acc": float(balanced_accuracy_score(y[va], pv.argmax(1)))})
        print(f"epoch {ep + 1:02d} val macro AUROC {auc:.4f}  bal.acc {hist[-1]['val_balanced_acc']:.3f}", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), ROOT / "results" / "best.pt")

    model.load_state_dict(torch.load(ROOT / "results" / "best.pt", map_location=dev))
    p = predict(model, X[te])  # test split touched once
    yt = y[te]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        b = rng.integers(0, len(te), len(te))
        if len(set(yt[b])) == len(CLASSES):
            boots.append(macro_auc(yt[b], p[b]))
    mel = CLASSES.index("mel")
    res = {
        "best_val_macro_auc": best,
        "best_epoch": max(hist, key=lambda h: h["val_macro_auc"])["epoch"],
        "test_macro_auc": macro_auc(yt, p),
        "test_macro_auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "test_balanced_acc": float(balanced_accuracy_score(yt, p.argmax(1))),
        "test_accuracy": float((p.argmax(1) == yt).mean()),
        "test_auc_per_class": {c: float(roc_auc_score(yt == i, p[:, i])) for i, c in enumerate(CLASSES)},
        "melanoma_recall_at_argmax": float(((p.argmax(1) == mel) & (yt == mel)).sum() / (yt == mel).sum()),
        "test_class_counts": {c: int((yt == i).sum()) for i, c in enumerate(CLASSES)},
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "history": hist,
    }
    np.save(ROOT / "results" / "test_probs.npy", p)
    np.save(ROOT / "results" / "test_idx.npy", te)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()

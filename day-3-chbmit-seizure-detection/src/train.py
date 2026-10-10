"""Train the 1D CNN on raw windows. SPLIT=patient (default) or SPLIT=random (leaky comparison)."""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from cache import load_cache, make_split
from data import ROOT
from model import EEGNet1D, normalise

SEED, EPOCHS, BS, LR = 42, int(os.environ.get("EPOCHS", 15)), 256, 2e-3
MODE = os.environ.get("SPLIT", "patient")
SUF = os.environ.get("TAG", "")  # e.g. "_hn" for the hard-negative model
HARD = bool(os.environ.get("HARD"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def batch(X, idx):
    return normalise(torch.from_numpy(X[np.sort(idx)].astype(np.float32)).to(dev))


def predict(model, X, idx):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for i in range(0, len(idx), 1024):
            out.append(torch.sigmoid(model(batch(X, idx[i:i + 1024])).float()).cpu())
    return torch.cat(out).numpy()


def wl_metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "prevalence": float(y.mean()), "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    res_dir = ROOT / "results"; res_dir.mkdir(exist_ok=True)
    X, y, c = load_cache()
    tr, va, te = (np.flatnonzero(m) for m in make_split(c, MODE))
    if HARD:  # windows the v1 model got wrong on TRAIN recordings (see mine.py); always label 0
        from data import CACHE, TRAIN
        hx = [np.load(CACHE / f"hard_{c}.npz")["X"] for c in TRAIN if (CACHE / f"hard_{c}.npz").exists()]
        hx = np.concatenate(hx)
        n0 = len(X)
        X = np.concatenate([X, hx]); y = np.concatenate([y, np.zeros(len(hx), y.dtype)])
        tr = np.concatenate([tr, np.arange(n0, len(X))])
        print(f"added {len(hx)} hard negatives", flush=True)
    print(f"mode={MODE} train={len(tr)} (pos {y[tr].sum()}) val={len(va)} test={len(te)}", flush=True)

    model = EEGNet1D().to(dev)
    pos_w = torch.tensor(np.sqrt((1 - y[tr].mean()) / y[tr].mean()), dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
    steps = EPOCHS * (len(tr) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.2)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    best, hist = -1.0, []
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = np.sort(perm[i:i + BS])
            xb, yb = batch(X, idx), torch.from_numpy(y[idx].astype(np.float32)).to(dev)
            drop = (torch.rand(xb.size(0), xb.size(1), 1, device=dev) < 0.05).float()  # random channel dropout
            xb = xb * (1 - drop) + 0.05 * torch.randn_like(xb)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        m = wl_metrics(y[va], predict(model, X, va))  # selection on validation windows only
        hist.append({"epoch": ep + 1, **m})
        print(f"epoch {ep + 1:02d} val AUPRC {m['auprc']:.4f} AUROC {m['auroc']:.4f}", flush=True)
        if m["auprc"] > best:
            best = m["auprc"]
            torch.save(model.state_dict(), res_dir / f"cnn_{MODE}{SUF}.pt")

    model.load_state_dict(torch.load(res_dir / f"cnn_{MODE}{SUF}.pt", map_location=dev))
    out = {"mode": MODE, "model": "cnn", "val_best_auprc": best,
           "best_epoch": max(hist, key=lambda h: h["auprc"])["epoch"], "test": wl_metrics(y[te], predict(model, X, te)),
           "history": hist}
    (res_dir / f"windows_cnn_{MODE}{SUF}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()

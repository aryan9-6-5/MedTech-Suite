import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from data import CLASSES, ROOT, load_meta, load_signals, splits, standardize
from model import ECGResNet

SEED, EPOCHS, BS = 42, 40, 128
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            out.append(torch.sigmoid(model(x[i:i + 512].to(dev))).cpu())
    return torch.cat(out).numpy()


def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])]))


def main():
    df, y = load_meta()
    tr, va, te = splits(df)
    x = standardize(load_signals(df), tr)
    X = torch.from_numpy(x).permute(0, 2, 1).contiguous()  # (N, 12, 1000)
    Y = torch.from_numpy(y)
    loader = DataLoader(TensorDataset(X[tr], Y[tr]), BS, shuffle=True, drop_last=True)

    model = ECGResNet().to(dev)
    pos = Y[tr].mean(0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).sqrt().to(dev))
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * len(loader))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    best, hist = 0.0, []
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            xb = xb * (1 + 0.1 * torch.randn(xb.size(0), 1, 1, device=dev))  # amplitude jitter
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        auc = macro_auc(y[va], predict(model, X[va]))  # model selection on fold 9 only
        hist.append({"epoch": ep + 1, "val_macro_auc": auc})
        print(f"epoch {ep + 1:02d} val macro AUC {auc:.4f}", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), ROOT / "results" / "best.pt")

    model.load_state_dict(torch.load(ROOT / "results" / "best.pt"))
    p = predict(model, X[te])
    np.save(ROOT / "results" / "test_probs.npy", p)
    np.save(ROOT / "results" / "test_idx.npy", np.where(te)[0])
    res = {
        "best_val_macro_auc": best,
        "test_macro_auc": macro_auc(y[te], p),
        "test_auc_per_class": {c: float(roc_auc_score(y[te][:, i], p[:, i])) for i, c in enumerate(CLASSES)},
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "history": hist,
    }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()

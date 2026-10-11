"""Experiment 2: train and apply a CNN with an optional baseline-relative input (variant 'rel' vs plain 'z')."""
import numpy as np
import torch
import torch.nn as nn

from data import baseline_scale, window_baseline, windows
from model import EEGNet1D, normalise

dev = "cuda" if torch.cuda.is_available() else "cpu"


def make_input(x, b, rel):
    """x (B,18,1024) microvolts, b (B,18) baseline RMS. 'z': per-window z-score only (as in experiment 1).
    'rel': also the same window divided by the patient's own baseline, so amplitude changes are kept."""
    z = normalise(x)
    if not rel:
        return z
    r = ((x - x.mean(-1, keepdim=True)) / (b[:, :, None] + 1e-3)).clamp(-20, 20)
    return torch.cat([z, r], 1)


def fit(X, y, b, tr, rel, epochs=12, seed=42, bs=256, lr=2e-3):
    """Fixed-epoch training, no model selection (so nothing is tuned on held-out data)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = EEGNet1D(36 if rel else 18).to(dev)
    pos_w = torch.tensor(np.sqrt((1 - y[tr].mean()) / y[tr].mean()), dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (len(tr) // bs), pct_start=0.2)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = np.sort(perm[i:i + bs])
            xb = torch.from_numpy(X[idx].astype(np.float32)).to(dev)
            bb = torch.from_numpy(b[idx]).to(dev)
            xb = xb * (1 - (torch.rand(xb.size(0), xb.size(1), 1, device=dev) < 0.05).float())  # channel dropout
            inp = make_input(xb, bb, rel)
            inp = inp + 0.05 * torch.randn_like(inp)
            yb = torch.from_numpy(y[idx].astype(np.float32)).to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(inp), yb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return model.eval()


def score_signal(model, rel, x):
    """(18, N) filtered recording -> one probability per second (hop 1 s), no sampling."""
    w = windows(x)
    b = window_baseline(baseline_scale(x), len(w))
    out = []
    for i in range(0, len(w), 512):
        xb = torch.from_numpy(np.ascontiguousarray(w[i:i + 512])).to(dev)
        bb = torch.from_numpy(np.ascontiguousarray(b[i:i + 512])).to(dev)
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(make_input(xb, bb, rel)).float()).cpu().numpy())
    return np.concatenate(out)

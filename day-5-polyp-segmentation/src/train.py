"""Train a polyp U-Net for a fixed number of epochs. ENC=resnet34|convnext, SEED=int. No model selection, nothing tuned on the external sets.

Saves sigmoid outputs (uint8, 0-255) for the internal val, internal test and both external sets to results/preds_{ENC}_s{SEED}.npz.
"""
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import ROOT, load, verify_external
from model import Net, augment, normalise

ENC = os.environ.get("ENC", "resnet34")
SEED = int(os.environ.get("SEED", 42))
EPOCHS, BS, LR = int(os.environ.get("EPOCHS", 40)), 16, 3e-4
dev = "cuda" if torch.cuda.is_available() else "cpu"


def to_x(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).permute(0, 3, 1, 2).float() / 255


def to_m(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).unsqueeze(1).float()


def loss_fn(logit, m):
    bce = F.binary_cross_entropy_with_logits(logit, m)
    p = torch.sigmoid(logit)
    inter, tot = (p * m).sum((1, 2, 3)), p.sum((1, 2, 3)) + m.sum((1, 2, 3))
    return bce + (1 - (2 * inter + 1) / (tot + 1)).mean()


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 32):
        with torch.autocast(dev, enabled=dev == "cuda"):
            p = torch.sigmoid(model(normalise(to_x(X[i:i + 32]))).float())
        out.append((p[:, 0] * 255).round().byte().cpu().numpy())
    return np.concatenate(out)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    R = ROOT / "results"; R.mkdir(exist_ok=True)
    d = load()
    verify_external()                                   # external sets must still match manifest.json
    part = d["part_kvasir"]
    tr, va, te = (np.flatnonzero(part == k) for k in (0, 1, 2))
    print(f"enc={ENC} seed={SEED} train={len(tr)} val={len(va)} test={len(te)} cvc={len(d['X_cvc'])} etis={len(d['X_etis'])}", flush=True)

    model = Net(ENC).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=0.01)
    steps = EPOCHS * (len(tr) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    rng = np.random.default_rng(SEED)
    Xk, Mk = d["X_kvasir"], d["M_kvasir"]
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(len(tr) // BS):
            idx = np.sort(order[i * BS:(i + 1) * BS])
            x, m = augment(to_x(Xk[idx]), to_m(Mk[idx]))
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(normalise(x)).float(), m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{EPOCHS} loss {tot / (len(tr) // BS):.4f} ({time.time() - t0:.0f}s)", flush=True)

    np.savez(R / f"preds_{ENC}_s{SEED}.npz", val=predict(model, Xk[va]), kvasir_test=predict(model, Xk[te]),
             cvc=predict(model, d["X_cvc"]), etis=predict(model, d["X_etis"]))
    print("saved predictions", flush=True)


if __name__ == "__main__":
    main()

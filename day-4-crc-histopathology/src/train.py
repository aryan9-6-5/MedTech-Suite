"""Fine-tune ConvNeXt-Tiny for a fixed number of epochs. ARM=A|B, SEED=int. No model selection, nothing tuned on the external set.

Saves softmax outputs for the internal validation, internal test and external sets to results/preds_{ARM}_s{SEED}.npz,
plus the weights to results/model_{ARM}_s{SEED}.pt.
"""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ROOT, load
from model import augment, build_model, normalise

ARM = os.environ.get("ARM", "A")
SEED = int(os.environ.get("SEED", 42))
EPOCHS, BS, LR = int(os.environ.get("EPOCHS", 6)), 64, 3e-4
dev = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(xb):
    return torch.from_numpy(np.ascontiguousarray(xb)).to(dev).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def predict(model, X, idx=None, feats=False):
    model.eval()
    idx = np.arange(len(X)) if idx is None else idx
    out, emb = [], []
    for i in range(0, len(idx), 256):
        xb = normalise(to_tensor(X[np.sort(idx[i:i + 256])]))
        with torch.autocast(dev, enabled=dev == "cuda"):
            z = model.avgpool(model.features(xb))   # (B, 768, 1, 1)
            logits = model.classifier(z)
        if feats:
            emb.append(z.flatten(1).float().cpu())
        out.append(F.softmax(logits.float(), 1).cpu())
    return (torch.cat(out).numpy(), torch.cat(emb).numpy()) if feats else torch.cat(out).numpy()


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    R = ROOT / "results"; R.mkdir(exist_ok=True)
    d = load()
    part, y = d["int_part"], d["int_y"]
    tr, va, te = (np.flatnonzero(part == k) for k in (0, 1, 2))
    print(f"arm={ARM} seed={SEED} train={len(tr)} val={len(va)} internal-test={len(te)} external={len(d['ext_y'])}", flush=True)

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=0.05)
    steps = EPOCHS * (len(tr) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = np.sort(perm[i:i + BS])
            xb = normalise(augment(to_tensor(d["int_x"][idx]), ARM))
            yb = torch.from_numpy(y[idx]).to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        pv = predict(model, d["int_x"], va)
        print(f"epoch {ep + 1}/{EPOCHS}  internal-val acc {(pv.argmax(1) == y[np.sort(va)]).mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), R / f"model_{ARM}_s{SEED}.pt")
    np.savez_compressed(R / f"preds_{ARM}_s{SEED}.npz", val=predict(model, d["int_x"], va), val_y=y[np.sort(va)],
                        test=predict(model, d["int_x"], te), test_y=y[np.sort(te)],
                        ext=predict(model, d["ext_x"]), ext_y=d["ext_y"])
    print("saved", flush=True)


if __name__ == "__main__":
    main()

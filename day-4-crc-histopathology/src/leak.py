"""How close is each test patch to the training set? Nearest-neighbour cosine similarity in the fine-tuned model's embedding space.

The internal test patches were drawn at random from the same 86 slides as the training patches, so they should sit
much closer to training patches than the external patches (new patients) do.
"""
import json

import numpy as np
import torch

from data import ROOT, load
from model import build_model
from train import dev, predict

R = ROOT / "results"


def nn_sim(train_emb, q, chunk=2048):
    t = torch.from_numpy(train_emb).to(dev)
    t = t / t.norm(dim=1, keepdim=True)
    out = []
    for i in range(0, len(q), chunk):
        x = torch.from_numpy(q[i:i + chunk]).to(dev)
        x = x / x.norm(dim=1, keepdim=True)
        out.append((x @ t.T).max(1).values.cpu())
    return torch.cat(out).numpy()


def main():
    d = load()
    part = d["int_part"]
    tr, te = np.flatnonzero(part == 0), np.flatnonzero(part == 2)
    m = build_model(pretrained=False).to(dev)
    m.load_state_dict(torch.load(R / "model_A_s42.pt", map_location=dev))
    _, e_tr = predict(m, d["int_x"], tr, feats=True)
    _, e_te = predict(m, d["int_x"], te, feats=True)
    _, e_ex = predict(m, d["ext_x"], None, feats=True)
    s_int, s_ext = nn_sim(e_tr, e_te), nn_sim(e_tr, e_ex)
    out = {k: {"n": int(len(s)), "median": float(np.median(s)), "p10": float(np.percentile(s, 10)),
               "frac_above_0.95": float((s > 0.95).mean()), "frac_above_0.90": float((s > 0.90).mean())}
           for k, s in (("internal_test", s_int), ("external", s_ext))}
    np.savez_compressed(R / "nn_similarity.npz", internal=s_int, external=s_ext)
    (R / "leakage.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

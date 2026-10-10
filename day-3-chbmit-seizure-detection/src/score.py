"""Score every validation and test recording end to end, one probability per second, no sampling."""
import json

import joblib
import numpy as np
import torch

from data import DATA, ROOT, TEST, VAL, parse_summary, read_edf, window_end_times, windows
from features import features
from model import EEGNet1D, normalise

dev = "cuda" if torch.cuda.is_available() else "cpu"
R = ROOT / "results"
OUT = R / "scores"


def main(mode="patient", cases=VAL + TEST):
    OUT.mkdir(parents=True, exist_ok=True)
    cnn = EEGNet1D().to(dev)
    cnn.load_state_dict(torch.load(R / f"cnn_{mode}.pt", map_location=dev))
    cnn.eval()
    gb = joblib.load(R / f"baseline_{mode}.joblib")
    for case in cases:
        if not (DATA / case / f"{case}-summary.txt").exists():
            print("WARNING: no data for", case, flush=True)
            continue
        for fname, szs in parse_summary(case).items():
            dst = OUT / f"{case}__{fname}.npz"
            p = DATA / case / fname
            if dst.exists() or not p.exists():
                continue
            x = read_edf(p)
            if x is None:
                continue
            w = windows(x)
            pc, pb = [], []
            for i in range(0, len(w), 512):
                chunk = np.ascontiguousarray(w[i:i + 512])
                with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
                    pc.append(torch.sigmoid(cnn(normalise(torch.from_numpy(chunk).to(dev))).float()).cpu().numpy())
                pb.append(gb.predict_proba(features(chunk))[:, 1])
            np.savez_compressed(dst, cnn=np.concatenate(pc).astype(np.float16), base=np.concatenate(pb).astype(np.float16),
                                duration=x.shape[1] / 256, seizures=np.array(szs, dtype=float).reshape(-1, 2))
            print(case, fname, f"{len(w)} windows, {len(szs)} seizures", flush=True)


if __name__ == "__main__":
    main()

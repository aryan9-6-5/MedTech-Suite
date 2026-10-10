"""Hard-negative mining. Run the v1 CNN over every TRAIN recording and keep its worst false positives.

Only training patients are touched. Validation and test patients are never mined.
"""
import numpy as np
import torch

from cache import NEAR
from data import CACHE, DATA, FS, TRAIN, WIN, parse_summary, read_edf, window_end_times, windows
from data import ROOT
from model import EEGNet1D, normalise

dev = "cuda" if torch.cuda.is_available() else "cpu"
PER_FILE, SPACING, MIN_PROB = 25, 10, 0.5  # at most 25 per recording, at least 10 s apart, v1 prob >= 0.5


def main(cases=TRAIN):
    cnn = EEGNet1D().to(dev)
    cnn.load_state_dict(torch.load(ROOT / "results" / "cnn_patient.pt", map_location=dev))
    cnn.eval()
    for case in cases:
        if not (DATA / case / f"{case}-summary.txt").exists():
            print("WARNING: no data for", case, flush=True)
            continue
        out = CACHE / f"hard_{case}.npz"
        if out.exists():
            continue
        picked = []
        for fname, szs in parse_summary(case).items():
            p = DATA / case / fname
            if not p.exists():
                continue
            x = read_edf(p)
            if x is None:
                continue
            w = windows(x)
            center = window_end_times(len(w)) - WIN / FS / 2
            near = np.zeros(len(w), bool)
            for on, off in szs:
                near |= (center >= on - NEAR) & (center < off + NEAR)
            probs = []
            for i in range(0, len(w), 512):
                with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
                    probs.append(torch.sigmoid(cnn(normalise(torch.from_numpy(np.ascontiguousarray(w[i:i + 512])).to(dev))).float()).cpu().numpy())
            probs = np.concatenate(probs)
            cand = np.flatnonzero((~near) & (probs >= MIN_PROB))
            kept = []
            for i in cand[np.argsort(-probs[cand])]:
                if all(abs(i - k) >= SPACING for k in kept):
                    kept.append(i)
                    if len(kept) >= PER_FILE:
                        break
            if kept:
                picked.append(w[np.sort(kept)].astype(np.float16))
            print(case, fname, "hard negatives", len(kept), flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(out, X=np.concatenate(picked) if picked else np.zeros((0, 18, WIN), np.float16))


if __name__ == "__main__":
    main()

"""Window cache: every seizure window plus sparse seizure-free background windows, per case."""
import json
import sys
import zlib

import numpy as np

from data import CACHE, CASES, DATA, FS, TEST, TRAIN, VAL, WIN, parse_summary, read_edf, window_end_times, windows

NEAR = 60  # seconds around a seizure excluded from background sampling (ambiguous pre/post-ictal)
BG_STRIDE = 60  # one background window per minute of recording


def build_case(case):
    out = CACHE / f"{case}.npz"
    if out.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    X, Y, T, meta = [], [], [], {}
    for fname, szs in parse_summary(case).items():
        p = DATA / case / fname
        if not p.exists():
            meta[fname] = {"status": "missing", "seizures": szs}
            continue
        x = read_edf(p)
        if x is None:
            meta[fname] = {"status": "channels", "seizures": szs}
            continue
        w = windows(x)
        center = window_end_times(len(w)) - WIN / FS / 2
        pos = np.zeros(len(w), bool)
        near = np.zeros(len(w), bool)
        for on, off in szs:
            pos |= (center >= on) & (center < off)
            near |= (center >= on - NEAR) & (center < off + NEAR)
        rng = np.random.default_rng(zlib.crc32(fname.encode()))
        neg = np.flatnonzero(~near)[int(rng.integers(BG_STRIDE))::BG_STRIDE]
        idx = np.concatenate([np.flatnonzero(pos), neg])
        X.append(w[idx].astype(np.float16))
        Y.append(pos[idx].astype(np.uint8))
        T.append(center[idx])
        meta[fname] = {"status": "ok", "duration_s": x.shape[1] / FS, "seizures": szs}
        print(case, fname, "pos", int(pos.sum()), "bg", len(neg), flush=True)
    np.savez(out, X=np.concatenate(X), y=np.concatenate(Y), t=np.concatenate(T))
    (CACHE / f"{case}.json").write_text(json.dumps(meta))


def load_cache(cases=CASES):
    X, y, c = [], [], []
    for i, case in enumerate(CASES):
        if case not in cases:
            continue
        if not (CACHE / f"{case}.npz").exists():
            print("WARNING: no cache for", case, flush=True)
            continue
        z = np.load(CACHE / f"{case}.npz")
        X.append(z["X"]); y.append(z["y"]); c.append(np.full(len(z["y"]), i))
    return np.concatenate(X), np.concatenate(y), np.concatenate(c)


def make_split(case_idx, mode, seed=42):
    """'patient': whole cases held out (fixed TRAIN/VAL/TEST lists). 'random': ignores patient (leaky)."""
    if mode == "patient":
        names = np.array(CASES)[case_idx]
        return (np.isin(names, TRAIN), np.isin(names, VAL), np.isin(names, TEST))
    perm = np.random.default_rng(seed).permutation(len(case_idx))
    n = len(perm)
    m = np.zeros((3, n), bool)
    m[0, perm[: int(.7 * n)]] = m[1, perm[int(.7 * n): int(.85 * n)]] = m[2, perm[int(.85 * n):]] = True
    return tuple(m)


if __name__ == "__main__":
    for case in (sys.argv[1:] or CASES):
        build_case(case)

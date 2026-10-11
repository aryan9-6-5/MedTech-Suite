"""Experiment-2 window cache: same sampling as cache.py, plus the per-window baseline vector."""
import json
import sys
import zlib

import numpy as np

from cache import BG_STRIDE, NEAR
from data import CACHE, CASES, DATA, FS, WIN, baseline_scale, parse_summary, read_edf, window_baseline, window_end_times, windows

CACHE2 = CACHE.parent / "cache2"


def build_case(case):
    out = CACHE2 / f"{case}.npz"
    if out.exists():
        return
    CACHE2.mkdir(parents=True, exist_ok=True)
    X, Y, B = [], [], []
    for fname, szs in parse_summary(case).items():
        p = DATA / case / fname
        x = read_edf(p) if p.exists() else None
        if x is None:
            continue
        w = windows(x)
        b = window_baseline(baseline_scale(x), len(w))
        center = window_end_times(len(w)) - WIN / FS / 2
        pos = np.zeros(len(w), bool)
        near = np.zeros(len(w), bool)
        for on, off in szs:
            pos |= (center >= on) & (center < off)
            near |= (center >= on - NEAR) & (center < off + NEAR)
        rng = np.random.default_rng(zlib.crc32(fname.encode()))
        neg = np.flatnonzero(~near)[int(rng.integers(BG_STRIDE))::BG_STRIDE]
        idx = np.concatenate([np.flatnonzero(pos), neg])
        X.append(w[idx].astype(np.float16)); Y.append(pos[idx].astype(np.uint8)); B.append(b[idx])
        print(case, fname, "pos", int(pos.sum()), "bg", len(neg), flush=True)
    np.savez(out, X=np.concatenate(X), y=np.concatenate(Y), b=np.concatenate(B))


def load2(cases=CASES):
    X, y, b, c = [], [], [], []
    for i, case in enumerate(CASES):
        if case in cases and (CACHE2 / f"{case}.npz").exists():
            z = np.load(CACHE2 / f"{case}.npz")
            X.append(z["X"]); y.append(z["y"]); b.append(z["b"]); c.append(np.full(len(z["y"]), i))
    return np.concatenate(X), np.concatenate(y), np.concatenate(b), np.concatenate(c)


if __name__ == "__main__":
    for case in (sys.argv[1:] or CASES):
        build_case(case)

"""Hand-crafted features for the baseline: relative band power, log total power, line length."""
import numpy as np

from data import FS, WIN

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
_freqs = np.fft.rfftfreq(WIN, 1 / FS)
_masks = [(_freqs >= lo) & (_freqs < hi) for lo, hi in BANDS]
_hann = np.hanning(WIN).astype(np.float32)


def features(w):
    """(n, 18, WIN) -> (n, 18 * 7). Relative powers make it less sensitive to per-patient amplitude."""
    w = np.asarray(w, dtype=np.float32)
    ll = np.log10(np.abs(np.diff(w, axis=-1)).sum(-1) + 1e-6)
    w = w - w.mean(-1, keepdims=True)
    p = np.abs(np.fft.rfft(w * _hann, axis=-1)) ** 2
    tot = p[..., (_freqs >= 0.5) & (_freqs < 45)].sum(-1) + 1e-9
    rel = [p[..., m].sum(-1) / tot for m in _masks]
    f = np.stack(rel + [np.log10(tot), ll], axis=-1)  # (n, 18, 7)
    return f.reshape(len(w), -1).astype(np.float32)

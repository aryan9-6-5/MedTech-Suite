"""CHB-MIT loading: summaries, 18-channel bipolar montage, filtering, 4 s windows at 1 s hop."""
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "chbmit"
CACHE = ROOT / "data" / "cache"
FS, WIN, HOP = 256, 1024, 256  # 4 s windows, 1 s hop

# Channels present in every case under the standard double-banana montage.
CH = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
      "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]

CASES = [f"chb{i:02d}" for i in range(1, 25)]
# Fixed before any training. Chosen for adequate seizure counts and a spread of difficulty.
TEST = ["chb03", "chb05", "chb10", "chb14", "chb20"]
VAL = ["chb08", "chb18", "chb23"]
TRAIN = [c for c in CASES if c not in TEST + VAL]


def subject(case):
    """chb21 is the same person as chb01 recorded 1.5 years later, so they share a subject."""
    return "chb01" if case == "chb21" else case


# A subject must never straddle train/val/test.
_sets = {name: {subject(c) for c in cs} for name, cs in (("train", TRAIN), ("val", VAL), ("test", TEST))}
assert not (_sets["train"] & _sets["val"]) and not (_sets["train"] & _sets["test"]) and not (_sets["val"] & _sets["test"])


def parse_summary(case):
    """-> {file_name: [(onset_s, offset_s), ...]} for every file listed in the summary."""
    text = (DATA / case / f"{case}-summary.txt").read_text(errors="ignore")
    out = {}
    for block in re.split(r"File Name:\s*", text)[1:]:
        name = block.split()[0]
        starts = [int(s) for s in re.findall(r"Seizure(?: \d+)? Start Time:\s*(\d+)", block)]
        ends = [int(s) for s in re.findall(r"Seizure(?: \d+)? End Time:\s*(\d+)", block)]
        assert len(starts) == len(ends), (case, name)
        out[name] = list(zip(starts, ends))
    return out


_sos = butter(4, [0.5, 45], btype="bandpass", fs=FS, output="sos")


def read_edf(path):
    """-> (18, N) float32 in microvolts, band-passed 0.5-45 Hz (zero phase), or None if channels are missing."""
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    names = {}
    for n in raw.ch_names:
        key = re.sub(r"-[01]$", "", n.upper().strip()) if n.upper().startswith("T8-P8") else n.upper().strip()
        names.setdefault(key, n)  # first occurrence wins (T8-P8 appears twice in some cases)
    if any(c not in names for c in CH):
        return None
    x = raw.get_data(picks=[names[c] for c in CH]).astype(np.float32) * 1e6
    return sosfiltfilt(_sos, x, axis=1).astype(np.float32)


def windows(x):
    """(18, N) -> (n, 18, WIN) view, hop = 1 s."""
    v = np.lib.stride_tricks.sliding_window_view(x, WIN, axis=1)[:, ::HOP]
    return v.transpose(1, 0, 2)


def window_end_times(n):
    """Time (s) at which each window is complete; alarms are stamped here so detection is causal."""
    return np.arange(n) * (HOP / FS) + WIN / FS

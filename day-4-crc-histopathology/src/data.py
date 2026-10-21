"""NCT-CRC-HE-100K (train) and CRC-VAL-HE-7K (external) from the Hugging Face mirror, balanced subsample, uint8 cache.

Source: Kather et al., Zenodo 10.5281/zenodo.1214456, CC BY 4.0. The mirror is checked against the class counts
published on Zenodo, so a silently different copy fails loudly.
"""
import io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
REPO = "1aurent/NCT-CRC-HE"
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
NAMES = {"ADI": "adipose", "BACK": "background", "DEB": "debris", "LYM": "lymphocytes", "MUC": "mucus",
         "MUS": "smooth muscle", "NORM": "normal mucosa", "STR": "cancer stroma", "TUM": "tumour epithelium"}
# Class counts listed in the Zenodo archives' file lists (same order as CLASSES).
PUBLISHED = {
    "NCT_CRC_HE_100K": [10407, 10566, 11512, 11557, 8896, 13536, 8763, 10446, 14317],
    "CRC_VAL_HE_7K": [1338, 847, 339, 634, 1035, 592, 741, 421, 1233],
}
N_TRAIN, N_VAL, N_TEST = 2500, 250, 500  # per class, drawn at random from the 100K set (internal, patch-level)
SEED = 42


def download():
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", local_dir=DATA / "hf",
                      allow_patterns=["data/NCT_CRC_HE_100K-*", "data/CRC_VAL_HE_7K-*"])


def shards(split):
    return sorted((DATA / "hf" / "data").glob(f"{split}-*.parquet"))


def read_labels(split, verify=True):
    import pyarrow.parquet as pq

    labels = [np.asarray(pq.read_table(f, columns=["label"])["label"].to_numpy(zero_copy_only=False)) for f in shards(split)]
    lens = [len(a) for a in labels]
    y = np.concatenate(labels).astype(np.int64)
    counts = np.bincount(y, minlength=len(CLASSES)).tolist()
    if verify:
        assert counts == PUBLISHED[split], f"{split}: mirror counts {counts} differ from published {PUBLISHED[split]}"
    return y, lens


def _decode(b):
    from PIL import Image

    im = Image.open(io.BytesIO(b)).convert("RGB")
    assert im.size == (224, 224), im.size
    return np.asarray(im)


def _extract(split, rows, out):
    """Fill `out[i]` with the image of global row `rows[i]` (rows sorted ascending)."""
    import pyarrow.parquet as pq

    pos = {int(r): i for i, r in enumerate(rows)}
    start = 0
    for f in shards(split):
        n = pq.ParquetFile(f).metadata.num_rows
        sel = [r - start for r in rows if start <= r < start + n]
        if sel:
            t = pq.read_table(f, columns=["image"]).take(sel)
            for r, rec in zip(sel, t["image"].to_pylist()):
                out[pos[r + start]] = _decode(rec["bytes"])
        start += n
        print(f"  {split} {f.name}: {len(sel)} patches", flush=True)


def build_cache(verify=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "done").exists():
        return
    rng = np.random.default_rng(SEED)
    y100, _ = read_labels("NCT_CRC_HE_100K", verify)
    sel, part = [], []
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(y100 == c))[: N_TRAIN + N_VAL + N_TEST]
        for k, (a, b) in enumerate([(0, N_TRAIN), (N_TRAIN, N_TRAIN + N_VAL), (N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST)]):
            sel += idx[a:b].tolist(); part += [k] * len(idx[a:b])
    order = np.argsort(sel)
    sel = np.array(sel)[order]; part = np.array(part)[order]  # 0 train, 1 val, 2 internal test
    x = np.lib.format.open_memmap(CACHE / "int_x.npy", mode="w+", dtype=np.uint8, shape=(len(sel), 224, 224, 3))
    _extract("NCT_CRC_HE_100K", sel.tolist(), x); x.flush()
    np.save(CACHE / "int_y.npy", y100[sel]); np.save(CACHE / "int_part.npy", part)
    yv, _ = read_labels("CRC_VAL_HE_7K", verify)
    xv = np.lib.format.open_memmap(CACHE / "ext_x.npy", mode="w+", dtype=np.uint8, shape=(len(yv), 224, 224, 3))
    _extract("CRC_VAL_HE_7K", list(range(len(yv))), xv); xv.flush()
    np.save(CACHE / "ext_y.npy", yv)
    (CACHE / "done").write_text("ok")


def load():
    """-> dict of memory-mapped images and label arrays."""
    d = {k: np.load(CACHE / f"{k}.npy", mmap_mode="r" if k.endswith("_x") else None)
         for k in ("int_x", "int_y", "int_part", "ext_x", "ext_y")}
    return d


if __name__ == "__main__":
    download()
    build_cache()

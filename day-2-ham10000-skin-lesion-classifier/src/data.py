"""HAM10000 loading: 7-class dermoscopy, lesion-grouped splits, cached 192x256 images."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NAMES = {
    "akiec": "actinic keratosis", "bcc": "basal cell carcinoma", "bkl": "benign keratosis",
    "df": "dermatofibroma", "mel": "melanoma", "nv": "melanocytic nevus", "vasc": "vascular lesion",
}
H, W = 192, 256  # keeps the 4:3 aspect of the originals


def load_meta():
    df = pd.read_csv(DATA / "HAM10000_metadata.tab", sep="\t")
    df["y"] = df.dx.map({c: i for i, c in enumerate(CLASSES)})
    return df


def make_splits(df, seed=42, mode="grouped"):
    """70/15/15 split grouped by lesion_id.

    HAM10000 holds several photos of the same lesion. A random image-level split puts
    near-duplicates on both sides and inflates every metric, so whole lesions stay together.
    """
    fold = np.zeros(len(df), dtype=int)
    if mode == "random":  # deliberately leaky: ignores lesion_id. Only for the leakage comparison.
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_splits=20, shuffle=True, random_state=seed).split(df, df.y)
    else:
        splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed).split(df, df.y, df.lesion_id)
    for k, (_, idx) in enumerate(splitter):
        fold[idx] = k
    split = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
    return split


def load_images(df):
    cache = DATA / "images_192x256.npy"
    if cache.exists():
        x = np.load(cache)
        if x.shape[0] == len(df):
            return x
    want = {f"{i}.jpg": n for n, i in enumerate(df.image_id)}
    x = np.zeros((len(df), H, W, 3), dtype=np.uint8)
    seen = 0
    for z in sorted(DATA.glob("HAM10000_images_part_*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base in want:
                    img = Image.open(io.BytesIO(zf.read(name))).convert("RGB").resize((W, H), Image.BILINEAR)
                    x[want[base]] = np.asarray(img)
                    seen += 1
    assert seen == len(df), f"found {seen} of {len(df)} images"
    np.save(cache, x)
    return x

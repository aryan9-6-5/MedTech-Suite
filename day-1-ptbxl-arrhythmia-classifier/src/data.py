"""PTB-XL loading: 100 Hz records, 5 diagnostic superclasses, official strat folds."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_meta():
    df = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(DATA / "scp_statements.csv", index_col=0)
    agg = agg[agg.diagnostic == 1]

    def to_super(codes):
        return sorted({agg.loc[c].diagnostic_class for c in codes if c in agg.index})

    df["superclass"] = df.scp_codes.apply(to_super)
    df = df[df.superclass.apply(len) > 0]  # drop records with no diagnostic label
    y = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for i, labs in enumerate(df.superclass):
        for lab in labs:
            y[i, CLASSES.index(lab)] = 1
    return df, y


def load_signals(df):
    cache = DATA / "x100.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(df):
        return np.load(cache)
    x = np.stack([wfdb.rdsamp(str(DATA / f))[0] for f in df.filename_lr]).astype(np.float32)
    np.save(cache, x)
    return x


def splits(df):
    f = df.strat_fold.values
    return f <= 8, f == 9, f == 10  # train / val / test (official recommendation)


def standardize(x, train_mask):
    mu = x[train_mask].mean(axis=(0, 1), keepdims=True)
    sd = x[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    return (x - mu) / sd

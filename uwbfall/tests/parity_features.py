"""Parity of uwbfall features against the team caches (run after `python -m uwbfall extract --sources val,train`).

1. extractor keys vs /home/user/data_raw/feats/<sid>.npz (analysis/extract_features output)
2. design matrices (64 prototype + 48 band channels, labels, folds) vs work/E4_deep_model/cache/blocks.npz
Usage: PYTHONPATH=/home/user/challenge1 python3 -m uwbfall.tests.parity_features
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from uwbfall.config import load_config
from uwbfall.extract import get_features
from uwbfall.features import block_labels, design, fit_pca, stream_blocks
from uwbfall.paths import PKG
from uwbfall.sources import split_sources

cfg = load_config()
REF = "/home/user/challenge1/work/E4_deep_model/cache/blocks.npz"
ev = pd.read_csv(PKG / "streams_events.csv", dtype=str)
ok = True
blks = {}
for split in ("train", "val"):
    for s in split_sources(split):
        F = get_features(s, cfg)
        with np.load(f"/home/user/data_raw/feats/{s.sid}.npz") as z:
            for k in F:
                eq = np.array_equal(F[k], z[k]) and F[k].dtype == z[k].dtype
                ok &= eq
                if not eq:
                    print("MISMATCH extract", s.sid, k)
        blks[s.sid] = stream_blocks(F, cfg)
        s.release()
print("extractor keys bit-equal to team cache:", ok, flush=True)
tr = sorted(k for k in blks if k.startswith("train"))
va = sorted(k for k in blks if k.startswith("val"))
pca = fit_pca([blks[k]["amp"] for k in tr], cfg)
R = np.load(REF)
for split, ids in (("train", tr), ("val", va)):
    Xp, Xb, Y = [], [], []
    for k in ids:
        xp, xb, cols = design(blks[k], pca, cfg)
        Xp.append(xp); Xb.append(xb); Y.append(block_labels(ev, k, blks[k]["n"])[0].astype(np.int8))
    Xp, Xb, Y = np.vstack(Xp), np.vstack(Xb), np.concatenate(Y)
    r = dict(X_proto=np.array_equal(Xp, R[f"{split}_X_proto"]), X_band=np.array_equal(Xb, R[f"{split}_X_band"]),
             y=np.array_equal(Y, R[f"{split}_y"]), cols=list(cols) == list(R["proto_cols"]))
    if split == "train":
        sid = np.concatenate([np.full(blks[k]["n"], k) for k in ids])
        fold = np.full(len(Y), -1, np.int8)
        for i, (_a, b) in enumerate(GroupKFold(4).split(Xp, Y, sid)):
            fold[b] = i
        r["fold"] = np.array_equal(fold, R["train_fold"])
    if not r["X_proto"]:
        d = np.abs(Xp - R[f"{split}_X_proto"])
        r["X_proto_maxabs"] = float(np.nanmax(d))
    ok &= all(v for k, v in r.items() if not k.endswith("maxabs"))
    print(split, r, flush=True)
print("ALL PARITY OK" if ok else "PARITY FAILED")
sys.exit(0 if ok else 1)

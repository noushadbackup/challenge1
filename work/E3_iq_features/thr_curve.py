"""Threshold sensitivity (plateau check): train-OOF (rep-0) ranking F1 vs threshold for fixed smooth/min_dur/gap,
and the same curve on val from the saved val probabilities (DIAGNOSTIC ONLY - never used for selection).
Usage: python3 thr_curve.py TAG [TAG ...]   (tags with saved preds/oof_TAG.npy and preds/valprob_TAG.npy)"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import iq_experiment as X
from common import stream_ids
sids = stream_ids()
tr = X.P.build([s for s in sids if s.startswith("train")])
va = X.P.build([s for s in sids if s.startswith("val")])
import pandas as pd
tr = pd.concat([d for d, _ in tr], ignore_index=True)
va = pd.concat([d for d, _ in va], ignore_index=True)
for tag in sys.argv[1:]:
    oof = np.load(X.HERE / "preds" / f"oof_{tag}.npy")
    pv = np.load(X.HERE / "preds" / f"valprob_{tag}.npy")
    for sm, md, gp in [(5, 0.8, 0.2), (5, 0.8, 0.6)]:
        row = []
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            a = X.score(X.events_from(tr, oof, (thr, sm, md, gp)), "train")["ranking_score"]["f1_mean"]
            b = X.score(X.events_from(va, pv, (thr, sm, md, gp)), "val")["ranking_score"]["f1_mean"]
            row.append(f"{thr}:{a:.3f}/{b:.3f}")
        print(f"{tag:14s} sm={sm} md={md} gap={gp}  thr:trainOOF/val  " + "  ".join(row), flush=True)

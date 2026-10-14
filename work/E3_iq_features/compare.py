"""Paired stream-cluster bootstrap of train-OOF ranking F1 between arms (rep-0 GroupKFold OOF).

Usage: python3 compare.py BASE_TAG ARM_TAG [ARM_TAG ...]   (tags = results/<tag>.json names)
Resamples the 13 train streams with replacement (B=5000), recomputes micro F1 at 0.1/0.3/0.5
from the official per-stream tp/fp/fn, and reports mean delta, 90 % CI and P(delta > 0).
Caveat: train OOF is leaky (shared subjects / ADL files across streams), and 13 clusters is few.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(tag):
    r = json.load(open(HERE / "results" / f"{tag}.json"))
    ps = r["train_oof_per_stream"]
    sids = sorted(ps)
    return sids, np.array([[ps[s][th] for th in ("0.1", "0.3", "0.5")] for s in sids], float)  # (S, 3, 3)


def rank_f1(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return f1.mean(-1)


base = sys.argv[1]
sids, cb = load(base)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(sids), (5000, len(sids)))
for tag in sys.argv[2:]:
    s2, ca = load(tag)
    assert s2 == sids
    d = rank_f1(ca[idx].sum(1)) - rank_f1(cb[idx].sum(1))
    point = rank_f1(ca.sum(0)) - rank_f1(cb.sum(0))
    lo, hi = np.percentile(d, [5, 95])
    print(f"{tag:28s} vs {base}: delta={point:+.4f}  90%CI [{lo:+.4f}, {hi:+.4f}]  P(>0)={np.mean(d > 0):.2f}")

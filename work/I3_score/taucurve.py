"""Diagnostic only (NOT used for any selection): held-out ranking F1 as a function of the cut tau.

For the hit-score rows with the reference kept set (XHPN: SR-CV-OOF hit model, pooled bags, detector-ordered NMS)
and for the detector score (REF), score every SR-CV fold and the pooled OOF at each tau of the grid with the
official evaluator, averaged over the 3 bags. Shows how flat the curve is around the chosen taus and where each
fold's own optimum lies (the stability question of W5).

    python3 work/I3_score/taucurve.py  -> out/taucurve.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
from common import BAGS  # noqa: E402
from srcv import scoring as S  # noqa: E402

TAUS = np.round(np.arange(0.0, 0.81, 0.02), 2)


def curve(variant):
    pooled = np.zeros(len(TAUS))
    folds = np.zeros((4, len(TAUS)))
    for bag in BAGS:
        P = {k: pd.read_csv(HERE / "preds" / bag / variant / f"fold_{k}.csv") for k in range(4)}
        for j, tau in enumerate(TAUS):
            Q = {k: p[p.score >= tau] for k, p in P.items()}
            pooled[j] += S.score_pooled(Q)["ranking_score"]["f1_mean"] / len(BAGS)
            for k in range(4):
                folds[k, j] += S.score(Q[k], k)["ranking_score"]["f1_mean"] / len(BAGS)
    return pooled, folds


def main():
    out = {"taus": TAUS.tolist()}
    for v in ("REF", "XHPN"):
        pooled, folds = curve(v)
        out[v] = {"pooled": pooled.round(4).tolist(), "folds": folds.round(4).tolist(),
                  "argmax_pooled": float(TAUS[int(np.argmax(pooled))]),
                  "argmax_folds": [float(TAUS[int(np.argmax(f))]) for f in folds]}
        print(v, "pooled argmax", out[v]["argmax_pooled"], "max", pooled.max().round(4), "at0", pooled[0].round(4),
              "fold argmax", out[v]["argmax_folds"], flush=True)
        for tau in (0.3, 0.4, 0.46, 0.5, 0.6):
            j = int(np.argmin(np.abs(TAUS - tau)))
            print(f"  tau {tau}: pooled {pooled[j]:.4f} folds {folds[:, j].round(4).tolist()}")
    (HERE / "out" / "taucurve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

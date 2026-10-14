"""Boundary-refinement variants computed from derive.py's saved replay tables (no new replay runs).

    python3 work/R2_EVAL/variants.py <bag>[,<bag>...]
For each base (C3: blend candidates, C4: TCN candidates) writes preds/<bag>/<base><suffix>/{fold_k,val}.csv:
  R  (= C3/C4 itself, re-derived; must be byte-identical: consistency check)
  K  "clean crops": a crop counts only if its annotator proposal carries no review reason
     (start_not_visible / floor_not_visible / event_under_1s / radar_onset_disagree). Every scored GT fall was
     produced by propose_fall with an EMPTY reason list (reasons -> needs_review -> flagged -> ignored), so a crop
     whose replay raises a reason saw a different context than the annotator did. No clean agreeing crop -> fall
     back to the replay-B rule (all agreeing crops).                                        [pre-registered fix]
  KD same, but no clean agreeing crop -> keep the detector bounds.                           [diagnostic]
  G3 agreement gate 0.3 instead of 0.1.                                                      [diagnostic]
  J  B's 6 crops + 6 jittered-start crops (jitter.py: {loc,det,detq}{1.5,2.5}_4d), median of the agreeing ones.
  JK J with the clean-crop rule of K.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from srcv import core as C
from uwbfall.boundary import tiou, to_preds
from uwbfall.config import load_config

HERE = Path(__file__).resolve().parent
B = load_config(None)["boundary"]
CROPS = B["crops"]


JCROPS = ["loc1.5_4d", "loc2.5_4d", "det1.5_4d", "det2.5_4d", "detq1.5_4d", "detq2.5_4d"]


def refine_variant(D: pd.DataFrame, mode: str):
    crops = CROPS + JCROPS if mode.startswith("J") else CROPS
    mode = {"J": "R", "JK": "K"}.get(mode, mode)
    s0, e0 = D.t_start.values.astype(float), D.t_end.values.astype(float)
    S = np.vstack([D[f"{c}_s"].values for c in crops]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in crops]).astype(float)
    gate = 0.3 if mode == "G3" else B["gate"]
    ok = np.isfinite(S) & np.isfinite(E) & (E > S) & (tiou(S, E, s0, e0) >= gate)
    if mode in ("K", "KD"):
        clean = np.vstack([D[f"{c}_rev"].isna().values for c in crops])
        okc = ok & clean
        has = okc.sum(0) >= 1
        sel = np.where(has[None, :], okc, ok if mode == "K" else okc)
    else:
        sel = ok
    use = sel.sum(0) >= 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rs = np.nanmedian(np.where(sel, S, np.nan), 0)
        re_ = np.nanmedian(np.where(sel, E, np.nan), 0)
    s = np.where(use, rs, s0)
    e = np.where(use, re_, e0)
    e = np.maximum(e, s + B["dmin"])
    return s, e


def durations(t):
    if t == "official":
        from abc2027 import evaluate as E
        return E.stream_durations(C.PKG / "streams" / "val")
    return C.durations(int(t), "test")


def main():
    for bag in sys.argv[1].split(","):
        for base in ("C3", "C4"):
            modes = ["R", "K", "KD", "G3"]
            if all((HERE / "cache" / f"DJ_{bag}_{t}_{base}.csv").exists() for t in ["0", "1", "2", "3", "official"]):
                modes += ["J", "JK"]
            for mode in modes:
                name = base + ("" if mode == "R" else mode)
                d = HERE / "preds" / bag / (base + "_R" if mode == "R" else name)
                d.mkdir(parents=True, exist_ok=True)
                for t in ["0", "1", "2", "3", "official"]:
                    D = pd.read_csv(HERE / "cache" / f"D_{bag}_{t}_{base}.csv")
                    if mode.startswith("J"):
                        DJ = pd.read_csv(HERE / "cache" / f"DJ_{bag}_{t}_{base}.csv")
                        assert len(DJ) == len(D)
                        assert (DJ.t_start.values == D.t_start.values).all()
                        D = pd.concat([D, DJ[[c for c in DJ.columns if any(c.startswith(j + "_") for j in JCROPS)]]],
                                      axis=1)
                    s, e = refine_variant(D, mode)
                    p = to_preds(D, s, e, durations(t))
                    p.to_csv(d / ("val.csv" if t == "official" else f"fold_{t}.csv"), index=False)
            # consistency: R must equal the replay-B outputs of derive.py
            for f in ["fold_0.csv", "fold_1.csv", "fold_2.csv", "fold_3.csv", "val.csv"]:
                a = (HERE / "preds" / bag / base / f).read_bytes()
                b = (HERE / "preds" / bag / (base + "_R") / f).read_bytes()
                assert a == b, (bag, base, f)
        print(f"{bag}: variants written; R == derive outputs (byte-identical)")


if __name__ == "__main__":
    main()

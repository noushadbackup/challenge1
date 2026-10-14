"""Per-event mechanism analysis of the replay x detector interaction.

For every candidate of a replayed system (C3 = blend + replay, C4 = TCN + replay) we have, from derive.py's
replay table D: detector bounds, the 6 crop replays, the refined bounds and whether the replay was used.
Each candidate is attached to the ground-truth event (incl. ignored ones) it overlaps best (by detector-bound
tIoU, else replay-bound tIoU) and we record the tIoU before (detector bounds) and after (replay) refinement.

    python3 work/R2_EVAL/mech.py [bags]   -> out/mech_events.csv, prints transition tables
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from srcv import core as C
from uwbfall.boundary import tiou
from uwbfall.paths import PKG

HERE = Path(__file__).resolve().parent
CROPS = ["loc2_3d", "loc2_5d", "det2_3d", "det2_5d", "detq2_3d", "detq2_5d"]


def gt_for(t):
    if t == "official":
        g = pd.read_csv(PKG / "streams_events.csv", dtype={"ignore": str})
        g = g[g.stream_id.str.startswith("val_")]
    else:
        g = pd.read_csv(C.events_csv(int(t), "test"), dtype={"ignore": str})
    g["ign"] = g["ignore"].astype(str).str.lower().eq("true") | g["needs_review"].astype(str).str.lower().eq("true")
    return g


def features(D: pd.DataFrame) -> pd.DataFrame:
    s0, e0 = D.t_start.values, D.t_end.values
    S = np.vstack([D[f"{c}_s"].values for c in CROPS]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in CROPS]).astype(float)
    ag = tiou(S, E, s0, e0)
    ok = np.isfinite(S) & np.isfinite(E) & (E > S) & (ag >= 0.1)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        f = pd.DataFrame({
            "det_w": e0 - s0,
            "n_ok": ok.sum(0),
            "agree_med": np.nanmedian(np.where(ok, ag, np.nan), 0),
            "agree_max": np.nanmax(np.where(np.isfinite(ag), ag, np.nan), 0),
            "start_spread": np.nanstd(np.where(ok, S, np.nan), 0),
            "rep_w": D.r_e.values - D.r_s.values,
            "agree_ref": tiou(D.r_s.values, D.r_e.values, s0, e0),
        })
    return f


def attach(D: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    out = []
    for i, r in D.iterrows():
        gs = g[g.stream_id == r.stream_id]
        if len(gs):
            di = tiou(r.t_start, r.t_end, gs.t_start.values, gs.t_end.values)
            ri = tiou(r.r_s, r.r_e, gs.t_start.values, gs.t_end.values)
            j = int(np.argmax(np.maximum(di, ri)))
            out.append({"gt_id": gs.event_id.values[j], "gt_ign": bool(gs.ign.values[j]),
                        "gt_s": gs.t_start.values[j], "gt_e": gs.t_end.values[j],
                        "iou_det": float(di[j]), "iou_rep": float(ri[j])})
        else:
            out.append({"gt_id": None, "gt_ign": False, "gt_s": np.nan, "gt_e": np.nan, "iou_det": 0.0, "iou_rep": 0.0})
    return pd.DataFrame(out, index=D.index)


def main():
    bags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["uwb_default"]
    rows = []
    for bag in bags:
        for t in ["0", "1", "2", "3", "official"]:
            g = gt_for(t)
            for c in ["C3", "C4"]:
                f = HERE / "cache" / f"D_{bag}_{t}_{c}.csv"
                if not f.exists():
                    continue
                D = pd.read_csv(f)
                X = pd.concat([D[["stream_id", "t_start", "t_end", "score", "use", "r_s", "r_e"]], features(D),
                               attach(D, g)], axis=1)
                X["bag"], X["target"], X["system"] = bag, t, c
                rows.append(X)
    X = pd.concat(rows, ignore_index=True)
    (HERE / "out").mkdir(exist_ok=True)
    X.to_csv(HERE / "out" / "mech_events.csv", index=False)
    X = X[~X.gt_ign]
    for th in (0.3, 0.5):
        X[f"before@{th}"] = X.iou_det >= th
        X[f"after@{th}"] = X.iou_rep >= th
    for split, Xs in (("SR-CV", X[X.target != "official"]), ("val", X[X.target == "official"])):
        print(f"===== {split}")
        for c in ["C3", "C4"]:
            x = Xs[Xs.system == c]
            for th in (0.3, 0.5):
                b, a = x[f"before@{th}"], x[f"after@{th}"]
                print(f"{c} @{th}: n={len(x)} matched before {b.sum()} after {a.sum()} | gained {(~b & a).sum()} "
                      f"lost {(b & ~a).sum()} | replay used {x.use.sum()}")
            lost = x[(x["before@0.3"]) & (~x["after@0.3"])]
            kept = x[(x["before@0.3"]) & (x["after@0.3"]) & x.use]
            cols = ["det_w", "n_ok", "agree_med", "agree_ref", "start_spread", "rep_w"]
            print("  lost@0.3 medians:", lost[cols].median().round(3).to_dict(), "n", len(lost))
            print("  kept@0.3 (replayed) medians:", kept[cols].median().round(3).to_dict(), "n", len(kept))


if __name__ == "__main__":
    main()

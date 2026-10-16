"""Headroom diagnostic: which evaluable falls have no matching final row, and what do their segments look like?

For every target and part (train = bundle's own training OOF, test = held-out fold / val) list
  * every evaluable GT fall: best tIoU of any final row, the junction segment holding its midpoint, that segment's
    P(fall clip) (gate), max gated / raw blend inside the segment, whether any final row overlaps the segment;
  * every segment without a final row: gate, max p, whether it holds an evaluable fall (= recoverable) or not.

    python3 work/R3_recover/diag.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import srcv
from srcv import core as C

R = Path("/home/user/data_raw/r3/recover")
HERE = Path(__file__).resolve().parent
STEP = 0.2


def tiou(a0, a1, b0, b1):
    i = np.maximum(0, np.minimum(a1, b1) - np.maximum(a0, b0))
    u = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(u > 0, i / np.where(u > 0, u, 1), 0.0)


def gt_for(target: str, part: str) -> pd.DataFrame:
    if target == "official_train":
        e = C.real_events("train" if part == "train" else "val")
    else:
        e = C.events(int(target.split("_")[1]), part)
    e = e.copy()
    e["t_start"] = e.t_start.astype(float)
    e["t_end"] = e.t_end.astype(float)
    ign = e["ignore"].astype(str).str.lower().eq("true")
    # official --ignore-flagged (also what the bundles' in-process OOF evaluator uses): low quality / needs_review
    ign |= e.quality.astype(str).eq("low") | e.needs_review.astype(str).str.lower().eq("true")
    e["evaluable"] = ~ign
    return e


def load(target: str, part: str):
    with open(R / target / f"{part}.pkl", "rb") as f:
        d = pickle.load(f)
    rows = d["oof_final"] if part == "train" else d["pred"]
    return d, rows


def tables(target: str, part: str):
    d, rows = load(target, part)
    gt = gt_for(target, part)
    ev, sg = [], []
    for sid, S in d["streams"].items():
        p, g, segs = S["p"], S["gate"], S["segs"]
        r = rows[rows.stream_id == sid]
        G = gt[gt.stream_id == sid]
        seg_has_row = np.zeros(len(segs), bool)
        for a, b in [(x.t_start, x.t_end) for x in r.itertuples()]:
            seg_has_row |= (segs[:, 0] * STEP < b) & (segs[:, 1] * STEP > a)
        seg_fall = np.zeros(len(segs), bool)
        seg_fall_any = np.zeros(len(segs), bool)
        for e in G.itertuples():
            m = 0.5 * (e.t_start + e.t_end)
            j = int(np.searchsorted(segs[:, 1] * STEP, m, side="right"))
            j = min(j, len(segs) - 1)
            best = float(tiou(r.t_start.values, r.t_end.values, e.t_start, e.t_end).max()) if len(r) else 0.0
            a, b = segs[j]
            seg_fall_any[j] = True
            if e.evaluable:
                seg_fall[j] = True
            ev.append(dict(target=target, part=part, stream_id=sid, t_start=e.t_start, t_end=e.t_end,
                           fall_type=getattr(e, "fall_type", ""), evaluable=e.evaluable, best_tiou=best,
                           seg=j, seg_a=a * STEP, seg_b=b * STEP, gate=float(g[a]), pmax_seg=float(p[a:b].max()),
                           pmax_ev=float(p[int(e.t_start / STEP):int(np.ceil(e.t_end / STEP)) + 1].max()),
                           seg_has_row=bool(seg_has_row[j])))
        for j, (a, b) in enumerate(segs):
            sg.append(dict(target=target, part=part, stream_id=sid, seg=j, a=a * STEP, b=b * STEP, len_s=(b - a) * STEP,
                           gate=float(g[a]), pmax=float(p[a:b].max()), has_row=bool(seg_has_row[j]),
                           fall=bool(seg_fall[j]), fall_any=bool(seg_fall_any[j])))
    return pd.DataFrame(ev), pd.DataFrame(sg)


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or [t for t in ["fold_0", "fold_1", "fold_2", "fold_3", "official_train"]
                               if (R / t / "test.pkl").exists()]
    EV, SG = [], []
    for t in targets:
        for part in ("train", "test"):
            e, s = tables(t, part)
            EV.append(e); SG.append(s)
    EV, SG = pd.concat(EV), pd.concat(SG)
    (HERE / "diag").mkdir(exist_ok=True)
    EV.to_csv(HERE / "diag" / "events.csv", index=False)
    SG.to_csv(HERE / "diag" / "segments.csv.gz", index=False)
    pd.set_option("display.width", 220)
    for part in ("train", "test"):
        e = EV[(EV.part == part) & EV.evaluable]
        miss = e[e.best_tiou < 0.3]
        print(f"\n=== {part}: {len(e)} evaluable falls, {len(miss)} with best tIoU < 0.3; "
              f"{(miss.best_tiou < 0.1).sum()} < 0.1; of misses {(~miss.seg_has_row).sum()} in a segment w/o row")
        print(miss[["target", "stream_id", "t_start", "t_end", "fall_type", "best_tiou", "gate", "pmax_seg", "pmax_ev",
                    "seg_has_row", "seg_a", "seg_b"]].round(3).to_string())
        s = SG[(SG.part == part) & ~SG.has_row]
        print(f"segments without a row: {len(s)}; with evaluable fall {s.fall.sum()}")
        for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            q = s[s.gate >= tau]
            print(f"  gate>={tau}: n={len(q)} fall={q.fall.sum()} nonfall={(~q.fall_any).sum()}  "
                  f"| and pmax>=0.1: n={(q.pmax >= .1).sum()} fall={(q.fall & (q.pmax >= .1)).sum()}"
                  f" | pmax>=0.2: n={(q.pmax >= .2).sum()} fall={(q.fall & (q.pmax >= .2)).sum()}")

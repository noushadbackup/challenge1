"""Per fall-type / per-subject event analysis of SR-CV predictions (evaluable GT only).

For each evaluable GT event: best tIoU of any predicted row (and whether it is matched at theta by the official
evaluator's greedy matching). hit@theta per fall type, and recall of Test_P3_3.

    python3 work/I2_boundaries/pertype.py <pred root> <variant> [bags]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BAGS  # noqa: E402
import srcv  # noqa: E402
from abc2027 import evaluate as EV  # noqa: E402


def gt_all():
    e = pd.concat([srcv.events(k, "test") for k in range(4)], ignore_index=True)
    e["ignore"] = e.ignore.astype(str).str.lower() == "true"
    return e


def match_flags(pred: pd.DataFrame, gt: pd.DataFrame, theta: float) -> np.ndarray:
    """Official greedy matching (score order, ignore flagged GT): matched flag per evaluable GT row."""
    out = np.zeros(len(gt), bool)
    for sid, g in gt.groupby("stream_id"):
        p = pred[pred.stream_id == sid].sort_values("score", ascending=False, kind="stable")
        used = set()
        ev = [(i, r.t_start, r.t_end) for i, r in zip(g.index, g.itertuples()) if not r.ignore]
        for r in p.itertuples():
            best, bi = theta, None
            for i, a, b in ev:
                if i in used:
                    continue
                inter = max(0, min(b, r.t_end) - max(a, r.t_start)); uni = max(b, r.t_end) - min(a, r.t_start)
                v = inter / uni if uni > 0 else 0
                if v >= best:
                    best, bi = v, i
            if bi is not None:
                used.add(bi)
        for i in used:
            out[gt.index.get_loc(i)] = True
    return out


def best_tiou(pred, gt):
    bt = np.zeros(len(gt))
    for j, r in enumerate(gt.itertuples()):
        p = pred[pred.stream_id == r.stream_id]
        if len(p):
            inter = np.maximum(0, np.minimum(p.t_end, r.t_end) - np.maximum(p.t_start, r.t_start))
            uni = np.maximum(p.t_end, r.t_end) - np.minimum(p.t_start, r.t_start)
            bt[j] = (inter / uni).max()
    return bt


def table(root, variant, bags):
    gt = gt_all()
    ev = gt[~gt.ignore].reset_index(drop=True)
    rows = []
    for b in bags:
        pred = pd.concat([pd.read_csv(Path(root) / b / variant / f"fold_{k}.csv") for k in range(4)])
        d = ev.copy()
        d["bag"] = b
        d["best"] = best_tiou(pred, ev)
        for th in (0.1, 0.3, 0.5, 0.7):
            d[f"m{th}"] = match_flags(pred, gt, th)[~gt.ignore.values]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    root, var = sys.argv[1], sys.argv[2]
    bags = sys.argv[3].split(",") if len(sys.argv) > 3 else BAGS
    D = table(root, var, bags)
    g = D.groupby("fall_type")
    print(f"== {var}: matched rate per fall type (3-bag mean), n = evaluable events")
    print(pd.DataFrame({"n": g.size() / len(bags), **{f"m@{t}": g[f"m{t}"].mean().round(3) for t in (0.1, 0.3, 0.5, 0.7)},
                        "med best tIoU": g.best.median().round(3)}))
    p = D[D.subject == "Test_P3_3"]
    print(f"Test_P3_3: n={len(p) // len(bags)}, recall @.1/.3/.5 = " +
          "/".join(f"{p[f'm{t}'].mean():.3f}" for t in (0.1, 0.3, 0.5)))
    print(D.groupby("dataset")[["m0.1", "m0.3", "m0.5"]].mean().round(3))

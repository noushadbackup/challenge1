"""Train-only diagnostic for the crop-aggregation rule (median vs largest-cluster with tIoU thr): per target and bag,
mean over theta .1/.3/.5 of the hit rate of the refined bounds of train-OOF candidates vs unflagged training GT
(each candidate scored against its best GT; candidates without GT >= 0.1 skipped).
    python3 work/I2_boundaries/aggdiag.py <targets>
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BAGS, HERE, REF_CFG, target_sources  # noqa: E402
from sel import unflagged  # noqa: E402
from uwbfall.boundary import refine, tiou  # noqa: E402

VARIANTS = {"median": {}, **{f"cl{x}": {"agg": "cluster", "cluster_iou": x} for x in (0.5, 0.7, 0.8, 0.9)},
            **{f"fc{x}": {"fallback": "contain", "contain_frac": x} for x in (0.6, 0.8, 1.0)}}


def hit(M, s, e, gt):
    out = []
    for i, sid in enumerate(M.stream_id.values):
        g = gt[gt.stream_id == sid]
        if not len(g):
            continue
        det = tiou(M.t_start.values[i], M.t_end.values[i], g.t_start.values, g.t_end.values)
        if det.max() < 0.1:
            continue
        j = int(np.argmax(det))
        out.append(float(tiou(s[i], e[i], g.t_start.values[j], g.t_end.values[j])))
    out = np.array(out)
    return float(np.mean([(out >= th).mean() for th in (0.1, 0.3, 0.5)])), float(np.mean(out >= 0.5)), len(out)


res = {}
for t in sys.argv[1].split(","):
    _tr, evtr, _te, _ = target_sources(t)
    gt = unflagged(evtr)
    R = pickle.load(open(HERE / "cache" / f"rep_{t}_train.pkl", "rb"))
    for bag in BAGS:
        M = R["cand"][bag].merge(R["D"], on=["stream_id", "t_start", "t_end"], how="left")
        for v, o in VARIANTS.items():
            s, e, _u = refine(M, {**REF_CFG["boundary"], **o})
            res[f"{t}|{bag}|{v}"] = hit(M, s, e, gt)
        print(t, bag, {v: round(res[f"{t}|{bag}|{v}"][0], 4) for v in VARIANTS}, flush=True)
f = HERE / "out" / "aggdiag.json"
old = json.loads(f.read_text()) if f.exists() else {}
old.update(res)
f.write_text(json.dumps(old, indent=1))

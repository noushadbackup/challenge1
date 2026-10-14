"""Boundary failure modes per fall type on SR-CV test (reference G7N4, 3 bags), from the replay internals.

Per evaluable GT event and bag:
  det_iou   best tIoU of a detector candidate (pre-replay)       final_iou  best tIoU of a final (replay+NMS) row
  pre_iou   best tIoU of a replayed row before NMS               mode       failure class at theta=0.5
Modes: ok | det_miss (no candidate with tIoU>=0.1) | nms_drop (a pre-NMS row hit, NMS removed it) |
       replay_worse (detector >= theta, replay < theta) | replay_fallback_bad (crops all rejected, detector < theta) |
       both_bad (detector < theta and replay < theta)
    python3 work/I2_boundaries/erb.py [theta]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BAGS, HERE, REF_CFG  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall.boundary import refine, tiou, to_preds  # noqa: E402
import srcv  # noqa: E402


def best(rows, a, b):
    if not len(rows):
        return 0.0, -1
    v = tiou(rows.t_start.values, rows.t_end.values, a, b)
    return float(v.max()), int(np.argmax(v))


def events_table(theta=0.5, targets=("0", "1", "2", "3")):
    out = []
    for t in targets:
        R = pickle.load(open(HERE / "cache" / f"rep_{t}_test.pkl", "rb"))
        k = int(t)
        gt = srcv.events(k, "test")
        gt = gt[gt.ignore.astype(str).str.lower() != "true"]
        dur = srcv.durations(k, "test")
        for bag in BAGS:
            M = R["cand"][bag].merge(R["D"], on=["stream_id", "t_start", "t_end"], how="left")
            st, en, use = refine(M, REF_CFG["boundary"])
            M["rs"], M["re"], M["use"] = st, en, use
            crops = REF_CFG["boundary"]["crops"]
            S = np.vstack([M[f"{c}_s"].values for c in crops]); E = np.vstack([M[f"{c}_e"].values for c in crops])
            M["n_ok"] = ((tiou(S, E, M.t_start.values, M.t_end.values) >= 0.3) & np.isfinite(S)).sum(0)
            M["spread_s"] = np.nanstd(np.where(np.isfinite(S), S, np.nan), 0)
            pre = to_preds(M, st, en, dur)
            fin = CL.nms_time(pre, 4.0)
            for r in gt.itertuples():
                c = M[M.stream_id == r.stream_id]
                di, j = best(c, r.t_start, r.t_end)
                pi, _ = best(pre[pre.stream_id == r.stream_id], r.t_start, r.t_end)
                fi, _ = best(fin[fin.stream_id == r.stream_id], r.t_start, r.t_end)
                d = dict(bag=bag, fold=k, stream_id=r.stream_id, t_start=r.t_start, t_end=r.t_end, dur=r.t_end - r.t_start,
                         fall_type=r.fall_type, subject=r.subject, dataset=r.dataset, det_iou=di, pre_iou=pi, final_iou=fi)
                if j >= 0:
                    cc = c.iloc[j]
                    d.update(cand_s=cc.t_start, cand_e=cc.t_end, rs=cc.rs, re=cc.re, use=bool(cc.use), n_ok=int(cc.n_ok),
                             spread=cc.spread_s, score=cc.score,
                             anchors=";".join(str(cc.get(f"{x}_anchor")) for x in crops))
                if fi >= theta:
                    mode = "ok"
                elif di < 0.1:
                    mode = "det_miss"
                elif pi >= theta:
                    mode = "nms_drop"
                elif di >= theta:
                    mode = "replay_worse"
                elif not d.get("use", False):
                    mode = "replay_fallback_bad"
                else:
                    mode = "both_bad"
                d["mode"] = mode
                out.append(d)
    return pd.DataFrame(out)


if __name__ == "__main__":
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    T = events_table(th)
    T.to_csv(HERE / "out" / f"erb_events_{th}.csv", index=False)
    print(f"== failure modes at theta={th} (events x bags; 3 bags)")
    print(pd.crosstab(T.fall_type, T["mode"], margins=True))
    c4 = T[(T.fall_type == "Chute04") & (T["mode"] != "ok")]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(c4[["bag", "stream_id", "t_start", "t_end", "subject", "det_iou", "final_iou", "cand_s", "cand_e", "rs", "re", "use",
              "n_ok", "spread", "score", "mode"]].round(2).to_string())

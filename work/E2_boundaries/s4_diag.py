"""Step 4 diagnostics (per event): matched GT, per-fall-type hit@0.5/0.7, replay failure rate,
signed start/end errors, and the signed error of the detected crop start vs the true junction.
Usage: python3 s4_diag.py <split> <refined_pred.csv> [<label>]
GT/composition are used here ONLY for diagnostics, never for building predictions."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
OUT = Path(__file__).resolve().parent / "out"
split, predp = sys.argv[1], sys.argv[2]
ev = pd.read_csv(PKG / "streams_events.csv")
ev = ev[ev.stream_id.str.startswith(split)].copy()
ev["flag"] = (ev.quality == "low") | ev.needs_review.astype(str).str.lower().eq("true")
evE = ev[~ev.flag]
co = pd.read_csv(PKG / "streams_composition.csv")
P = pd.read_csv(predp)
import os
R = pd.read_csv(os.environ.get("REPLAY_DIR", str(OUT)) + f"/replay_{split}.csv")


def best_iou(s, e, G):
    i = np.maximum(0, np.minimum(e, G.t_end.values) - np.maximum(s, G.t_start.values))
    u = np.maximum(e, G.t_end.values) - np.minimum(s, G.t_start.values)
    iou = i / u
    k = int(np.argmax(iou)) if len(iou) else -1
    return (iou[k], k) if k >= 0 else (0.0, -1)


rows = []
for g in evE.itertuples():
    Q = P[P.stream_id == g.stream_id]
    if len(Q) == 0:
        rows.append((g.fall_type, 0.0, np.nan, np.nan)); continue
    i = np.maximum(0, np.minimum(Q.t_end.values, g.t_end) - np.maximum(Q.t_start.values, g.t_start))
    u = np.maximum(Q.t_end.values, g.t_end) - np.minimum(Q.t_start.values, g.t_start)
    k = int(np.argmax(i / u)); q = Q.iloc[k]
    rows.append((g.fall_type, (i / u)[k], q.t_start - g.t_start, q.t_end - g.t_end))
D = pd.DataFrame(rows, columns=["type", "iou", "ds", "de"])
lab = sys.argv[3] if len(sys.argv) > 3 else predp
print(f"== {lab} ({split}, {len(D)} evaluable GT) best-IoU per GT")
T = D.groupby("type").agg(n=("iou", "size"), hit03=("iou", lambda x: (x >= .3).mean()), hit05=("iou", lambda x: (x >= .5).mean()),
                          hit07=("iou", lambda x: (x >= .7).mean()), med_ds=("ds", "median"), med_de=("de", "median"))
T.loc["ALL"] = [len(D), (D.iou >= .3).mean(), (D.iou >= .5).mean(), (D.iou >= .7).mean(), D.ds.median(), D.de.median()]
print(T.round(3).to_string())
h = D[D.iou >= 0.3]
print("signed start err (TP@0.3) p10/p50/p90: %s ; end err: %s" % (np.round(np.percentile(h.ds, [10, 50, 90]), 2), np.round(np.percentile(h.de, [10, 50, 90]), 2)))
# junction error: crop start (detected) vs true clean start of the fall segment containing the GT
fs = co[(co.kind == "fall") & co.stream_id.str.startswith(split)]
errs, errs_r = [], []
for r in R.itertuples():
    G = evE[(evE.stream_id == r.stream_id) & (evE.t_end > r.t_start) & (evE.t_start < r.t_end)]
    if len(G) == 0 or not np.isfinite(r.jprev):
        continue
    g = G.iloc[0]
    seg = fs[(fs.stream_id == r.stream_id) & (fs.t_start <= g.t_start) & (fs.t_end >= g.t_end)]
    if len(seg) == 0:
        continue
    clean = seg.t_start.iloc[0] + 0.2
    errs.append(r.jprev + 0.2 - clean); errs_r.append(r.jprev_ref + 0.15 - clean)
errs, errs_r = np.array(errs), np.array(errs_r)
for nm, x in (("block", errs), ("frame-refined", errs_r)):
    print(f"crop-start error vs true clean start [{nm}] n={len(x)}: |err|<=0.3 s {np.mean(np.abs(x) <= .3):.3f}, "
          f"p5/p50/p95 {np.round(np.percentile(x, [5, 50, 95]), 2)}, before true start (foreign content) {np.mean(x < -0.05):.3f}")

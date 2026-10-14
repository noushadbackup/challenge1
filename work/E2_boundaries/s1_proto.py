"""Step 1: reproduce the prototype 'full' detector exactly (same features, params, folds, grid)
and save (a) train OOF frame probabilities + OOF events with the train-selected post-processing,
(b) val frame probabilities + val events. These are the candidate events whose boundaries E2 refines.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/challenge1/analysis")
from prototype_detector import (build, add_fingerprint, cols_for, to_events, write_sub, official, STEP)
from common import stream_ids
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

OUT = Path(__file__).resolve().parent / "out"
t0 = time.time()
sids = stream_ids()
tr_ids = [s for s in sids if s.startswith("train")]
va_ids = [s for s in sids if s.startswith("val")]
tr, pca = add_fingerprint(build(tr_ids), None)
va, _ = add_fingerprint(build(va_ids), pca)
cols = cols_for(tr, "full")
params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
oof = np.zeros(len(tr)); fold = np.zeros(len(tr), int)
for k, (a, b) in enumerate(GroupKFold(4).split(tr, tr.y, tr.sid)):
    m = HistGradientBoostingClassifier(**params).fit(tr.iloc[a][cols], tr.y.iloc[a])
    oof[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]; fold[b] = k
print("oof done", time.time() - t0, flush=True)
grid = [(thr, sm, md, gp) for thr in (0.3, 0.4, 0.5, 0.6, 0.7) for sm in (1, 3, 5) for md in (0.4, 0.8) for gp in (0.2, 0.6)]
best = None
for g in grid:
    evs = [e for sid, d in tr.assign(p=oof).groupby("sid") for e in to_events(sid, d.t.values - STEP / 2, d.p.values, *g)]
    write_sub(evs, OUT / "tmp_train.csv")
    sc = official(OUT / "tmp_train.csv", "train", OUT / "tmp_train.json")["ranking_score"]["f1_mean"]
    if best is None or sc > best[0]:
        best = (sc, g)
print("best", best, time.time() - t0, flush=True)
evs = [e for sid, d in tr.assign(p=oof).groupby("sid") for e in to_events(sid, d.t.values - STEP / 2, d.p.values, *best[1])]
write_sub(evs, OUT / "train_oof_proto.csv")
model = HistGradientBoostingClassifier(**params).fit(tr[cols], tr.y)
pv = model.predict_proba(va[cols])[:, 1]
evs = [e for sid, d in va.assign(p=pv).groupby("sid") for e in to_events(sid, d.t.values - STEP / 2, d.p.values, *best[1])]
write_sub(evs, OUT / "val_proto.csv")
pd.DataFrame({"sid": tr.sid, "t": tr.t, "y": tr.y, "p": oof, "fold": fold}).to_parquet(OUT / "train_oof_frames.parquet") if False else \
    np.savez(OUT / "frames.npz", tr_sid=tr.sid.values.astype(str), tr_t=tr.t.values, tr_y=tr.y.values, tr_p=oof, tr_fold=fold,
             va_sid=va.sid.values.astype(str), va_t=va.t.values, va_y=va.y.values, va_p=pv)
json.dump({"train_oof_rank": best[0], "postproc": best[1]}, open(OUT / "proto_meta.json", "w"))
r = official(OUT / "train_oof_proto.csv", "train", OUT / "train_oof_proto.json")
print("train OOF", r["ranking_score"], flush=True)
r = official(OUT / "val_proto.csv", "val", OUT / "val_proto.json")
print("val", r["ranking_score"], time.time() - t0, flush=True)

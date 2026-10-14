"""Segment-classifier feature-set ablation. Selection criterion = SUBJECT-grouped OOF on the training streams of each
target (legit: training data only). Held-out test AUC is reported as a diagnostic, not used for selection.

    python3 work/I1_clip/segclf_ablate.py
"""
import json
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from uwbfall import clip as CL
from uwbfall.config import load_config
from srcv import core as C

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segclf_ablate_sets import sets  # noqa: E402

HERE = Path(__file__).resolve().parent
cfg = load_config({"clip": {"segclf": {"enabled": True}}})
PKGC = pd.read_csv(C.PKG / "streams_composition.csv")


def comp_for(t, part, sid):
    if t == "official":
        c = PKGC[PKGC.stream_id == sid].copy()
        c["subject"] = c.path.map(C.subject_of)
    else:
        c = C.composition(int(t), "train" if part == "train" else "test")
        c = c[c.stream_id == sid]
    return c


def subjects(T, t):
    out = []
    for sid, g in T.groupby("sid", sort=False):
        part = g.part.iloc[0]
        c = comp_for(t, part, sid)
        t0, t1 = g.a.values * 0.2, g.b.values * 0.2
        ov = np.minimum(c.t_end.values.astype(float)[None], t1[:, None]) - np.maximum(c.t_start.values.astype(float)[None], t0[:, None])
        out.append(pd.Series(c.subject.values[np.argmax(ov, 1)], index=g.index))
    return pd.concat(out).loc[T.index]


res = {}
TARGETS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "2", "3", "official"]
for t in TARGETS:
    P = pickle.load(open(HERE / "cache" / f"prep_{t}.pkl", "rb"))
    T = P["table"]
    T["subj"] = subjects(T, t)
    P["table"] = T
    pickle.dump(P, open(HERE / "cache" / f"prep_{t}.pkl", "wb"))
    tr, te = T[T.part == "train"], T[T.part == "test"]
    res[t] = {}
    for name, fs in sets(P["feats"]).items():
        oof = np.zeros((len(tr), 3))
        for a, b in GroupKFold(4).split(tr, tr.y, tr.subj):
            m = CL.lgbm_segclf(cfg).fit(tr.iloc[a][fs], tr.y.iloc[a])
            oof[b] = m.predict_proba(tr.iloc[b][fs])
        m = CL.lgbm_segclf(cfg).fit(tr[fs], tr.y)
        pt = m.predict_proba(te[fs])
        r = {"subjOOF_auc": roc_auc_score(tr.y == 1, oof[:, 1]), "subjOOF_ll": log_loss(tr.y, oof, labels=[0, 1, 2]),
             "test_auc": roc_auc_score(te.y == 1, pt[:, 1]), "test_ll": log_loss(te.y, pt, labels=[0, 1, 2])}
        res[t][name] = {k: round(float(v), 4) for k, v in r.items()}
        print(t, name, res[t][name], flush=True)
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / f"segclf_ablate_{'_'.join(TARGETS)}.json").write_text(json.dumps(res, indent=1))
df = pd.DataFrame({(t, n): v for t, d in res.items() for n, v in d.items()}).T
print(df.groupby(level=1).mean().round(4))

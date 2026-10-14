"""Per-block gates from the segment classifier with a fixed feature set:
train streams -> subject-grouped OOF P(fall clip) (GroupKFold(4) by segment subject; training data only),
test streams  -> final model trained on all training segments.
Stored in cache/prep_<t>.pkl under key <name> = {sid: (n,) array}.

    python3 work/I1_clip/segclf_final.py <feature_set> <name> [targets]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from uwbfall import clip as CL
from uwbfall.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segclf_ablate_sets import sets  # noqa: E402

HERE = Path(__file__).resolve().parent
cfg = load_config({"clip": {"segclf": {"enabled": True}}})
fsname, name = sys.argv[1], sys.argv[2]
targets = sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1", "2", "3", "official"]
for t in targets:
    f = HERE / "cache" / f"prep_{t}.pkl"
    P = pickle.load(open(f, "rb"))
    T = P["table"]
    fs = sets(P["feats"])[fsname]
    trm = (T.part == "train").values
    tr = T[trm]
    pf = np.zeros(len(T))
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(4).split(tr, tr.y, tr.subj):
        m = CL.lgbm_segclf(cfg).fit(tr.iloc[a][fs], tr.y.iloc[a])
        oof[b] = m.predict_proba(tr.iloc[b][fs])[:, 1]
    pf[trm] = oof
    m = CL.lgbm_segclf(cfg).fit(tr[fs], tr.y)
    pf[~trm] = m.predict_proba(T[~trm][fs])[:, 1]
    G = {}
    for sid, idx in T.groupby("sid", sort=False).groups.items():
        g = T.loc[idx]
        segs = np.stack([g.a.values, g.b.values], 1)
        G[sid] = CL.block_gate(segs, pf[np.asarray([T.index.get_loc(i) for i in idx])], P["n"][sid])
    P[name] = G
    P[name + "_meta"] = {"feature_set": fsname, "features": fs}
    pickle.dump(P, open(f, "wb"))
    print(t, name, fsname, len(fs), "features; mean gate train/test",
          round(float(np.mean(oof)), 3), round(float(np.mean(pf[~trm])), 3), flush=True)

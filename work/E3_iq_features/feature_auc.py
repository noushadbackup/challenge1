"""Per-feature AUC of the new I/Q features (train streams only).

Positives: 0.2 s blocks inside evaluable (unflagged) falls. Negatives: (a) fall-recording
margins (same segment, outside the fall), (b) ADL, (c) Autre/Empty. AUC is reported as
the raw Mann-Whitney AUC P(pos > neg) (values < 0.5 mean the feature is lower in falls), per dataset of the fall (D1/D2/D3 all + D3 unflagged = P3_3).
Usage: python3 feature_auc.py  -> results/feature_auc.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import iq_experiment as X  # noqa: E402
from common import auc, events, frame_labels, stream_ids  # noqa: E402

rows = []
for sid in [s for s in stream_ids() if s.startswith("train")]:
    n = len(np.load(X.FEATS / f"{sid}.npz")["prof_02"][0])
    f = {**X.iq_features(sid, n), **X.zone_static(sid, n)}
    f = {k: v for k, v in f.items() if "@" not in k}
    lab = frame_labels(sid, n, X.STEP)
    d = pd.DataFrame(f)
    d["kind"], d["label"], d["fall"], d["flag"] = lab["kind"], lab["label"], lab["fall"], lab["flagged"]
    ev = events()[lambda e: e.stream_id == sid]
    ds = np.full(n, "", object)
    for e in ev.itertuples():
        ds[(lab["t"] >= e.t_start) & (lab["t"] < e.t_end)] = e.dataset if hasattr(e, "dataset") else e.source_path.split("/")[1]
    d["ds"] = ds
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
print(D.ds.value_counts().to_dict())
feat = [c for c in D.columns if c.startswith(("iq_", "zst"))]
pos_all = D[D.fall & ~D.flag]
neg = {"margin": D[(D.kind == "fall") & ~D.fall], "adl": D[D.kind == "adl"], "autre_empty": D[D.kind.isin(["other", "autre", "empty"]) | D.label.isin(["Autre", "Empty"])]}
print({k: len(v) for k, v in neg.items()}, "pos", len(pos_all), pos_all.ds.value_counts().to_dict())
out = []
for c in feat:
    r = {"feature": c}
    for nk, nd in neg.items():
        for dsn, pd_ in [("all", pos_all)] + [(k, pos_all[pos_all.ds == k]) for k in sorted(pos_all.ds.unique())]:
            a = auc(pd_[c].values, nd[c].sample(min(len(nd), 20000), random_state=0).values)
            r[f"{nk}|{dsn}"] = round(a, 3)
    out.append(r)
O = pd.DataFrame(out)
O.to_csv(HERE / "results" / "feature_auc.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(O.to_string(index=False))

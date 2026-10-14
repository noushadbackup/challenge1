"""Reference: the prototype 'full' HGB (identical params / folds / grid), saving OOF and val probabilities."""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from lib import OUT, events_from_probs, load, official, summary, tune  # noqa: E402

D = load()
Xtr, ytr, fold = D["train_X_proto"], D["train_y"], D["train_fold"]
params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
oof = np.zeros(len(ytr))
for k in range(4):
    m = HistGradientBoostingClassifier(**params).fit(Xtr[fold != k], ytr[fold != k])
    oof[fold == k] = m.predict_proba(Xtr[fold == k])[:, 1]
pv = HistGradientBoostingClassifier(**params).fit(Xtr, ytr).predict_proba(D["val_X_proto"])[:, 1]
np.savez(OUT / "probs_hgb.npz", oof=oof, val=pv)
sc, g = tune(D["train_sid"], D["train_t"], oof)
r = official(events_from_probs(D["val_sid"], D["val_t"], pv, g), "val_hgb")
res = {"train_oof_rank_f1": round(sc, 4), "postproc": g, "val": summary(r)}
print(json.dumps(res))
json.dump(res, open(OUT / "res_hgb.json", "w"), indent=1)

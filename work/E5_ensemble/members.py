"""Train ensemble members with fold x seed bagging; save OOF (train) and bagged val probabilities.

Usage: python3 members.py <member> [<member> ...]
  members: hgb_proto (prototype reference, early_stopping=False), lgb_fused, xgb_fused, cat_fused,
           hgb_fused, lgb_radar0/1/2, lgb_physics
Folds: prototype GroupKFold(4) by stream (train only). OOF[k] = mean over seeds of fold-k models.
Val = mean over all fold x seed models (no refit, so the OOF-tuned threshold transfers).
"""
from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

from e5lib import OUT, frame, load, view_cols

SEEDS = (0, 1)
N_JOBS = 2


def make(model: str, seed: int):
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    if model == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                  n_jobs=N_JOBS, random_state=seed, deterministic=True, force_row_wise=True, verbose=-1)
    if model == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                                 min_child_weight=5, reg_lambda=1.0, tree_method="hist", n_jobs=N_JOBS, random_state=seed)
    if model == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, learning_rate=0.08, depth=6, l2_leaf_reg=3, rsm=0.7,
                                  thread_count=N_JOBS, random_seed=seed, verbose=0, allow_writing_files=False)
    raise ValueError(model)


def run(member: str) -> None:
    model, view = member.split("_", 1)
    d = load()
    tr, va = frame("train"), frame("val")
    cols = view_cols(tr, view)
    Xtr, Xva = tr[cols].values.astype(np.float32), va[cols].values.astype(np.float32)
    if model == "hgb" and view == "proto":  # keep the prototype's float64 inputs for the reference
        Xtr, Xva = tr[cols].values, va[cols].values
    y, fold = d["train_y"].astype(int), d["train_fold"]
    oof = np.zeros(len(y))
    pv = np.zeros(len(Xva))
    t0 = time.time()
    nfit = 0
    for k in range(4):
        a, b = fold != k, fold == k
        for s in SEEDS:
            m = make(model, s).fit(Xtr[a], y[a])
            oof[b] += m.predict_proba(Xtr[b])[:, 1] / len(SEEDS)
            pv += m.predict_proba(Xva)[:, 1]
            nfit += 1
    pv /= nfit
    np.savez_compressed(OUT / f"probs_{member}.npz", oof=oof.astype(np.float32), val=pv.astype(np.float32))
    from sklearn.metrics import roc_auc_score, average_precision_score
    info = {"member": member, "n_features": len(cols), "fits": nfit, "wall_s": round(time.time() - t0, 1),
            "oof_auc": round(roc_auc_score(y, oof), 5), "oof_ap": round(average_precision_score(y, oof), 4),
            "val_auc": round(roc_auc_score(d["val_y"], pv), 5), "val_ap": round(average_precision_score(d["val_y"], pv), 4),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    print(json.dumps(info), flush=True)
    with open(OUT / "members_log.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")


if __name__ == "__main__":
    for m in sys.argv[1:]:
        run(m)

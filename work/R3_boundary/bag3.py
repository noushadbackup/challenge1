"""3-bag robustness of wdet_cv: base vs wdet_cv for TCN bags 0/1/2 (bag 0 = final_norep; bags 1/2 = composite
norep bundles from bags.py, each with its own retuned base), seed-averaged promotion via srcv.promote(list, list).
    python3 bag3.py   -> results/wdet_cv_3bag.json"""
from common import *  # noqa
import json
import srcv
from uwbfall.scoring import summary

A, B, vA, vB = [], [], [], []
for bag in ("", "bag1", "bag2"):
    if bag:
        A.append({k: BIG / f"dump_{bag}" / f"fold_{k}" / "base_test.csv" for k in range(4)})
        vA.append(srcv.score_val(BIG / f"dump_{bag}" / "val" / "base_test.csv")["ranking_score"]["f1_mean"])
    else:
        A.append({k: base_csv(k) for k in range(4)})
        vA.append(srcv.score_val(base_csv("val"))["ranking_score"]["f1_mean"])
    d = HERE / "results" / ("wdet_cv" + (f"_{bag}" if bag else ""))
    B.append({k: d / f"fold_{k}.csv" for k in range(4)})
    vB.append(srcv.score_val(d / "val.csv")["ranking_score"]["f1_mean"])
pr = srcv.promote(A, B, val_A=float(np.mean(vA)), val_B=float(np.mean(vB)))
per_bag = []
for a, b, va, vb in zip(A, B, vA, vB):
    p = srcv.promote(a, b, val_A=va, val_B=vb)
    per_bag.append({k: p[k] for k in ("pooled_A", "pooled_B", "delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved",
                                      "val_A", "val_B", "promote")})
res = {"three_bag": pr, "per_bag": per_bag, "val_A": vA, "val_B": vB}
(HERE / "results" / "wdet_cv_3bag.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, default=float, indent=1))

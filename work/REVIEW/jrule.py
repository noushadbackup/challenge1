"""Junction-onset rule (data-team N5 / annotate ONSET_VISIBLE_S = 1.0): drop a final prediction whose start lies
within +-W s of a DETECTED junction peak (fold bundle threshold, features only). Applied after NMS to FINAL preds.
Writes preds to jr/<bag>/JR<W>/ and runs srcv.promote FINAL -> JR (3 bags) incl. official val."""
import sys, json, os
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np, pandas as pd
from srcv import scoring as S
P = "/home/user/challenge1/work/FINAL/preds"
J = json.load(open("detj.json"))
W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
name = f"JR{W}"
A, B, vA, vB = [], [], [], []
ndrop = 0
for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
    os.makedirs(f"jr/{bag}/{name}", exist_ok=True)
    a, b = {}, {}
    for part in [0, 1, 2, 3, "val"]:
        f = f"{P}/{bag}/FINAL/fold_{part}.csv" if part != "val" else f"{P}/{bag}/FINAL/val.csv"
        df = pd.read_csv(f)
        keep = []
        for _, r in df.iterrows():
            pk = np.array(J[r.stream_id])
            keep.append(not (len(pk) and np.min(np.abs(r.t_start - pk)) < W))
        out = df[np.array(keep)]
        ndrop += (~np.array(keep)).sum()
        o = f"jr/{bag}/{name}/" + (f"fold_{part}.csv" if part != "val" else "val.csv")
        out.to_csv(o, index=False)
        if part == "val":
            vA.append(f); vB.append(o)
        else:
            a[part] = f; b[part] = o
    A.append(a); B.append(b)
va = float(np.mean([S.score_val(x)["ranking_score"]["f1_mean"] for x in vA]))
vb = float(np.mean([S.score_val(x)["ranking_score"]["f1_mean"] for x in vB]))
r = S.promote(A, B, val_A=va, val_B=vb)
print(name, "rows dropped (3 bags, folds+val):", ndrop)
print(json.dumps({k: r[k] for k in ("pooled_A", "pooled_B", "delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved", "val_A", "val_B", "promote")}))
print([round(x["delta"], 4) for x in r["per_fold"]])
for bag, bb in zip(("uwb_default", "tcn_bag1", "tcn_bag2"), B):
    rp = S.score_pooled(bb); bt = rp["by_theta"]["0.3"]
    print("  ", bag, round(rp["ranking_score"]["f1_mean"], 4), (bt["tp"], bt["fp"], bt["fn"]))
json.dump(r, open(f"promote_FINAL_vs_{name}.json", "w"), indent=1)

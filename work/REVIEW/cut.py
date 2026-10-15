import sys
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np, pandas as pd
from srcv import scoring as S
P = "/home/user/challenge1/work/FINAL/preds"
for cut in (0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
    sr, va, fp = [], [], []
    for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
        d = {k: (lambda x: x[x.score >= cut])(pd.read_csv(f"{P}/{bag}/FINAL/fold_{k}.csv")) for k in range(4)}
        r = S.score_pooled(d); sr.append(r["ranking_score"]["f1_mean"]); fp.append((r["by_theta"]["0.3"]["tp"], r["by_theta"]["0.3"]["fp"]))
        v = pd.read_csv(f"{P}/{bag}/FINAL/val.csv"); va.append(S.score_val(v[v.score >= cut])["ranking_score"]["f1_mean"])
    print(f"score>={cut:.2f}  SR-CV {np.mean(sr):.4f} (bags {np.round(sr,4)})  TP/FP@0.3 {np.mean(fp,0).round(1)}  val {np.mean(va):.4f}")
v = pd.concat([pd.read_csv(f"{P}/{b}/FINAL/fold_{k}.csv") for b in ("uwb_default",) for k in range(4)])
print("score quantiles (bag0 SR-CV rows):", np.percentile(v.score, [0, 5, 10, 25, 50]).round(3), len(v))

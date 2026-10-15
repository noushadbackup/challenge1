"""Simulate the official test draw: 2 D1 + 2 D2 + 2 D3 fall subjects + 2 ADL subjects, sampled without replacement
from the SR-CV pool, using per-subject official TP/FP/FN counts (srcv.scoring.cluster_counts). Per-bag counts."""
import sys, json
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np, pandas as pd
from srcv import scoring as S
P = "/home/user/challenge1/work/FINAL/preds"
TH = ("0.1", "0.3", "0.5")
res = {}
for sysname in sys.argv[1].split(","):
    Xs = []
    for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
        cc = S.cluster_counts({k: f"{P}/{bag}/{sysname}/fold_{k}.csv" for k in range(4)})
        cc = cc[cc.theta.isin(TH)]
        t = cc.groupby(["subject", "theta"])[["tp", "fp", "fn"]].sum()
        subj = sorted(cc.subject.unique())
        Xs.append(np.stack([[t.loc[(s, th)].values for th in TH] for s in subj]).astype(float))
        if bag == "uwb_default":
            per = cc[cc.theta == "0.3"].groupby("subject")[["tp", "fp", "fn"]].sum()
            per.to_csv(f"persubj_{sysname}.csv")
    X = np.stack(Xs)  # (bag, subj, th, c)
    strata = {"D1": [i for i, s in enumerate(subj) if "_P1_" in s], "D2": [i for i, s in enumerate(subj) if "_P2_" in s],
              "D3": [i for i, s in enumerate(subj) if "_P3_" in s], "ADL": [i for i, s in enumerate(subj) if s.startswith("ADL_")]}
    rng = np.random.default_rng(0)
    B = 20000
    f = np.zeros(B); has33 = np.zeros(B, bool)
    i33 = subj.index("Test_P3_3")
    for b in range(B):
        idx = np.concatenate([rng.choice(v, 2, replace=False) for v in strata.values()])
        bag = rng.integers(0, 3)
        Sm = X[bag, idx].sum(0)
        tp, fp, fn = Sm[:, 0], Sm[:, 1], Sm[:, 2]
        f[b] = np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1))
        has33[b] = i33 in idx
    q = np.percentile(f, [2.5, 10, 25, 50, 75, 90, 97.5])
    res[sysname] = {"mean": f.mean(), "sd": f.std(), "q": q.tolist(), "mean_with_P3_3": f[has33].mean(), "mean_without": f[~has33].mean(),
                    "p_with": has33.mean()}
    print(sysname, "mean %.4f sd %.4f" % (f.mean(), f.std()), "q2.5/10/25/50/75/90/97.5", np.round(q, 3),
          "with P3_3 %.4f (p=%.2f) without %.4f" % (f[has33].mean(), has33.mean(), f[~has33].mean()))
json.dump(res, open("testdraw.json", "w"), indent=1)

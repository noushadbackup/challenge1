import sys
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np
from srcv import scoring as S
P = "/home/user/challenge1/work/FINAL/preds"; TH = ("0.1", "0.3", "0.5")
Xs = []
for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
    cc = S.cluster_counts({k: f"{P}/{bag}/FINAL/fold_{k}.csv" for k in range(4)}); cc = cc[cc.theta.isin(TH)]
    t = cc.groupby(["subject", "theta"])[["tp", "fp", "fn"]].sum(); subj = sorted(cc.subject.unique())
    Xs.append(np.stack([[t.loc[(s, th)].values for th in TH] for s in subj]).astype(float))
X = np.stack(Xs); st = [[i for i, s in enumerate(subj) if k in s] for k in ("_P1_", "_P2_", "_P3_", "ADL_")]
i12 = subj.index("ADL_12"); rng = np.random.default_rng(5); B = 20000; f = np.zeros(B); h = np.zeros(B, bool)
for b in range(B):
    idx = np.concatenate([rng.choice(v, 2, replace=False) for v in st]); S_ = X[rng.integers(0, 3), idx].sum(0)
    f[b] = np.mean(2 * S_[:, 0] / np.maximum(2 * S_[:, 0] + S_[:, 1] + S_[:, 2], 1)); h[b] = i12 in idx
print("with ADL_12 %.4f (p=%.2f)  without %.4f" % (f[h].mean(), h.mean(), f[~h].mean()))

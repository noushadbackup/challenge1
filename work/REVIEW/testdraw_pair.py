"""Paired test-draw: same subject draw and bag for C3 and FINAL; P(FINAL > C3), percentile of val 0.952."""
import sys
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np
from srcv import scoring as S
P = "/home/user/challenge1/work/FINAL/preds"; TH = ("0.1", "0.3", "0.5")
def tens(sysname):
    Xs = []
    for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
        cc = S.cluster_counts({k: f"{P}/{bag}/{sysname}/fold_{k}.csv" for k in range(4)}); cc = cc[cc.theta.isin(TH)]
        t = cc.groupby(["subject", "theta"])[["tp", "fp", "fn"]].sum(); subj = sorted(cc.subject.unique())
        Xs.append(np.stack([[t.loc[(s, th)].values for th in TH] for s in subj]).astype(float))
    return np.stack(Xs), subj
XF, subj = tens("FINAL"); XC, _ = tens("C3")
st = [[i for i, s in enumerate(subj) if k in s] for k in ("_P1_", "_P2_", "_P3_", "ADL_")]
rng = np.random.default_rng(1); B = 20000; f = np.zeros(B); c = np.zeros(B)
F1 = lambda S_: np.mean(2 * S_[:, 0] / np.maximum(2 * S_[:, 0] + S_[:, 1] + S_[:, 2], 1))
for b in range(B):
    idx = np.concatenate([rng.choice(v, 2, replace=False) for v in st]); g = rng.integers(0, 3)
    f[b] = F1(XF[g, idx].sum(0)); c[b] = F1(XC[g, idx].sum(0))
print("P(FINAL>C3) %.3f  P(FINAL>=C3) %.3f  mean diff %.4f  q5 diff %.4f" % ((f > c).mean(), (f >= c).mean(), (f - c).mean(), np.percentile(f - c, 5)))
print("percentile of 0.952 in FINAL draws: %.1f" % (100 * (f < 0.952).mean()))

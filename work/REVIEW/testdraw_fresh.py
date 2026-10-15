"""Test-draw simulation (2+2+2 fall + 2 ADL subjects) from per-subject counts of FINAL bag 0 on the FRESH streams,
and pooled with the 3 original bags (each replicate drawn with equal probability)."""
import sys, json
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np
from pathlib import Path
import srcv.core as C
from srcv import scoring as S
TH = ("0.1", "0.3", "0.5")
def tens(preds):
    cc = S.cluster_counts(preds); cc = cc[cc.theta.isin(TH)]
    t = cc.groupby(["subject", "theta"])[["tp", "fp", "fn"]].sum(); subj = sorted(cc.subject.unique())
    return np.stack([[t.loc[(s, th)].values for th in TH] for s in subj]).astype(float), subj
P = "/home/user/challenge1/work/FINAL/preds"
orig = [tens({k: f"{P}/{b}/FINAL/fold_{k}.csv" for k in range(4)}) for b in ("uwb_default", "tcn_bag1", "tcn_bag2")]
C.DATA = Path("/home/user/data_raw/review_fresh"); C._comp.cache_clear()
fresh = tens({k: f"/home/user/data_raw/review_fresh/preds/FINAL/fold_{k}.csv" for k in range(4)})
subj = fresh[1]; assert all(o[1] == subj for o in orig)
st = [[i for i, s in enumerate(subj) if k in s] for k in ("_P1_", "_P2_", "_P3_", "ADL_")]
F1 = lambda S_: np.mean(2 * S_[:, 0] / np.maximum(2 * S_[:, 0] + S_[:, 1] + S_[:, 2], 1))
out = {}
for name, reps in (("fresh_bag0", [fresh[0]]), ("orig3+fresh", [o[0] for o in orig] + [fresh[0]])):
    rng = np.random.default_rng(3); f = np.zeros(20000)
    for b in range(len(f)):
        idx = np.concatenate([rng.choice(v, 2, replace=False) for v in st]); X = reps[rng.integers(0, len(reps))]
        f[b] = F1(X[idx].sum(0))
    q = np.percentile(f, [2.5, 10, 25, 50, 75, 90, 97.5])
    out[name] = {"mean": f.mean(), "sd": f.std(), "q": q.tolist()}
    print(name, "mean %.4f sd %.4f" % (f.mean(), f.std()), np.round(q, 3))
json.dump(out, open("testdraw_fresh.json", "w"), indent=1)

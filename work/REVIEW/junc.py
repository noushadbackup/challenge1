"""Hindsight diagnostic: offset of each prediction start from the nearest TRUE junction (composition segment
starts), split by outcome @0.3 (TP / FP / ignored). Also: GT evaluable onset offsets. 3 bags, FINAL."""
import sys
sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
from pathlib import Path
from abc2027 import evaluate as E
import pandas as pd, numpy as np
R = Path("/home/user/data_raw/srcv"); P = Path("/home/user/challenge1/work/FINAL/preds")
sysname = sys.argv[1]
rows = []
gto = []
for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
    for k in range(4):
        gt = E.load_ground_truth(R / f"fold_{k}/test_events.csv", ignore_flagged=True)
        comp = pd.read_csv(R / f"fold_{k}/test_composition.csv")
        pr = pd.read_csv(P / bag / sysname / f"fold_{k}.csv")
        for sid, g in comp.groupby("stream_id"):
            J = np.sort(g.t_start.values[1:])  # junction = start of each non-first segment
            gs = [x for x in gt if x["stream_id"] == sid]
            if bag == "uwb_default":
                for e in gs:
                    if not e["ignore"]:
                        d = e["t_start"] - J[J <= e["t_start"] + 1e-9]
                        gto.append(d.min() if len(d) else 99)
            ps = [dict(r, t_alarm=None) for r in pr[pr.stream_id == sid][["stream_id","t_start","t_end","score"]].to_dict("records")]
            res = E._match_stream(gs, ps, 0.3)
            matched = {a for a, _, _ in res["pairs"]}
            ign = [x for x in gs if x["ignore"]]
            for i, p in enumerate(ps):
                dd = p["t_start"] - J
                d = dd[np.argmin(np.abs(dd))]
                out = "TP" if i in matched else ("IGN" if any(E._intervals_overlap(p["t_start"], p["t_end"], e["t_start"]-2, e["t_end"]+2) for e in ign) else "FP")
                rows.append((bag, k, out, d))
df = pd.DataFrame(rows, columns=["bag", "fold", "out", "d"])
gto = np.array(gto)
print("evaluable GT onset - preceding junction: min %.2f, q1%% %.2f, share<1.0 s %.3f" % (gto.min(), np.percentile(gto, 1), (gto < 1.0).mean()))
bins = [-99, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 99]
df["bin"] = pd.cut(df.d, bins)
print((df.groupby(["bin", "out"], observed=False).size().unstack(fill_value=0) / 3).round(1))
for lo, hi in [(-0.5, 0.8), (-0.5, 0.5), (-1.0, 0.8)]:
    m = (df.d > lo) & (df.d < hi)
    print(f"window ({lo},{hi}): per bag TP {(m & (df.out=='TP')).sum()/3:.1f}  FP {(m & (df.out=='FP')).sum()/3:.1f}")

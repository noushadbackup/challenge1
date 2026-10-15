"""Independent pooled re-score of SR-CV predictions using only the official evaluator functions.
GT: concatenation of the 4 fold test_events.csv; durations: last composition t_end per stream (=T/50)."""
import sys, json, csv, tempfile, os
from pathlib import Path
sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
from abc2027 import evaluate as E
import pandas as pd
R = Path("/home/user/data_raw/srcv")
P = Path("/home/user/challenge1/work/FINAL/preds")
gt, dur = [], {}
for k in range(4):
    gt += E.load_ground_truth(R / f"fold_{k}/test_events.csv", ignore_flagged=True)
    c = pd.read_csv(R / f"fold_{k}/test_composition.csv")
    for sid, g in c.groupby("stream_id"):
        dur[sid] = float(g.t_end.max())
def recs(files):
    df = pd.concat([pd.read_csv(f) for f in files])
    fd, p = tempfile.mkstemp(suffix=".csv"); os.close(fd); df.to_csv(p, index=False)
    r = E.validate_predictions(Path(p), dur); os.unlink(p); return r
out = {}
n_eval = sum(not g["ignore"] for g in gt)
print("evaluable falls", n_eval, "streams", len(dur), "hours", round(sum(dur.values())/3600, 3))
for sysname in ("C3", "FINAL"):
    for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
        r = E.evaluate(gt, recs([P / bag / sysname / f"fold_{k}.csv" for k in range(4)]), dur)
        bt = r["by_theta"]
        perfold = []
        for k in range(4):
            sub = {s: d for s, d in dur.items() if s.startswith(f"cv{k}_")}
            rk = E.evaluate([g for g in gt if g["stream_id"] in sub], recs([P / bag / sysname / f"fold_{k}.csv"]), sub)
            perfold.append(round(rk["ranking_score"]["f1_mean"], 4))
        out[(sysname, bag)] = r["ranking_score"]["f1_mean"]
        print(sysname, bag, round(r["ranking_score"]["f1_mean"], 4), [round(bt[t]["f1"], 3) for t in ("0.1","0.3","0.5","0.7")],
              (bt["0.3"]["tp"], bt["0.3"]["fp"], bt["0.3"]["fn"]), "FA/h", round(bt["0.3"]["false_alarms_per_hour"], 2),
              "medstart", round(bt["0.3"]["start_error_median_abs"], 3), "folds", perfold)
import numpy as np
for s in ("C3", "FINAL"):
    print(s, "3-bag mean", round(np.mean([v for (a, b), v in out.items() if a == s]), 4))

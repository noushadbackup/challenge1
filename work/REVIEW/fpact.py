import sys
sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
from pathlib import Path
from abc2027 import evaluate as E
import pandas as pd, numpy as np, collections
R = Path("/home/user/data_raw/srcv"); P = Path("/home/user/challenge1/work/FINAL/preds")
cnt = collections.Counter(); cntk = collections.Counter(); tot = collections.Counter()
for sysname in sys.argv[1:]:
  cnt = collections.Counter(); kinds = collections.Counter()
  for bag in ("uwb_default", "tcn_bag1", "tcn_bag2"):
    for k in range(4):
        gt = E.load_ground_truth(R / f"fold_{k}/test_events.csv", ignore_flagged=True)
        comp = pd.read_csv(R / f"fold_{k}/test_composition.csv")
        pr = pd.read_csv(P / bag / sysname / f"fold_{k}.csv")
        for sid, g in comp.groupby("stream_id"):
            gs = [x for x in gt if x["stream_id"] == sid]
            ps = [dict(r, t_alarm=None) for r in pr[pr.stream_id == sid][["stream_id","t_start","t_end","score"]].to_dict("records")]
            res = E._match_stream(gs, ps, 0.3)
            matched = {a for a, _, _ in res["pairs"]}
            ign = [x for x in gs if x["ignore"]]
            for i, p in enumerate(ps):
                if i in matched or any(E._intervals_overlap(p["t_start"], p["t_end"], e["t_start"]-2, e["t_end"]+2) for e in ign):
                    continue
                ov = np.minimum(g.t_end.values, p["t_end"]) - np.maximum(g.t_start.values, p["t_start"])
                j = int(np.argmax(ov)); row = g.iloc[j]
                cnt[(row.kind, row.label, row.subject)] += 1
                kinds[row.kind] += 1
  print(sysname, "FP@0.3 summed over 3 bags by segment kind:", dict(kinds))
  for (kd, lab, s), n in cnt.most_common(15): print("   ", kd, lab, s, n)

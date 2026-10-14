"""Post-hoc NMS variants on existing final prediction sets (no refit, no replay).

    python3 work/I1_clip/apply_nms.py <src_variant> <dst_variant> time:<gap> | segment[:max_span] [bags]
segment NMS uses the junction peaks of cache/prep_<target>.pkl.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

from uwbfall import clip as CL

HERE = Path(__file__).resolve().parent
src, dst, mode = sys.argv[1:4]
bags = sys.argv[4].split(",") if len(sys.argv) > 4 else ["uwb_default", "tcn_bag1", "tcn_bag2"]
for b in bags:
    for t, name in [("0", "fold_0"), ("1", "fold_1"), ("2", "fold_2"), ("3", "fold_3"), ("official", "val")]:
        df = pd.read_csv(HERE / "preds" / b / src / f"{name}.csv")
        if mode.startswith("time"):
            out = CL.nms_time(df, float(mode.split(":")[1]))
        else:
            P = pickle.load(open(HERE / "cache" / f"prep_{t}.pkl", "rb"))
            span = float(mode.split(":")[1]) if ":" in mode else 18.7
            out = CL.nms_segment(df, P["peaks"], span)
        d = HERE / "preds" / b / dst
        d.mkdir(parents=True, exist_ok=True)
        out.to_csv(d / f"{name}.csv", index=False)
    print(b, dst, "done")

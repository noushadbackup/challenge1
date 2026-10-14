"""FP / FN breakdown at theta 0.3 for val submissions (official matching code, in-process).

Usage: python3 fp_breakdown.py preds/val_full_r3.csv preds/val_full+iq_r3.csv
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
sys.path.insert(0, "/home/user/challenge1/analysis")
import pandas as pd  # noqa: E402
from abc2027 import evaluate as EV  # noqa: E402
from common import PKG, composition, events  # noqa: E402

gt = EV.load_ground_truth(PKG / "streams_events.csv", ignore_flagged=True)
dur = EV.stream_durations(PKG / "streams" / "val")
co = composition()
ev = events()
for path in sys.argv[1:]:
    preds = EV.validate_predictions(Path(path), dur)
    fps, fns = [], []
    for sid in sorted(dur):
        g = [e for e in gt if e["stream_id"] == sid]
        p = [x for x in preds if x["stream_id"] == sid]
        r = EV._match_stream(g, p, 0.3)
        matched = {a for a, _b, _c in r["pairs"]}
        mg = {b for _a, b, _c in r["pairs"]}
        ign = [e for e in g if e["ignore"]]
        for i, x in enumerate(p):
            if i in matched or any(EV._intervals_overlap(x["t_start"], x["t_end"], e["t_start"] - 2, e["t_end"] + 2) for e in ign):
                continue
            mid = 0.5 * (x["t_start"] + x["t_end"])
            seg = co[(co.stream_id == sid) & (co.t_start <= mid) & (co.t_end > mid)]
            fps.append({"sid": sid, "t0": x["t_start"], "t1": x["t_end"], "kind": seg.kind.iloc[0] if len(seg) else "?",
                        "label": seg.label.iloc[0] if len(seg) else "?"})
        for j, e in enumerate(r["scored"]):
            if j not in mg:
                row = ev[(ev.stream_id == sid) & (abs(ev.t_start - e["t_start"]) < 1e-3)].iloc[0]
                fns.append({"sid": sid, "t0": e["t_start"], "type": row.fall_type, "dataset": row.dataset, "subject": row.subject})
    print(f"== {path}: FP {len(fps)}, FN {len(fns)}")
    print(pd.DataFrame(fps).to_string(index=False) if fps else "no FP")
    print(pd.DataFrame(fns).to_string(index=False) if fns else "no FN")

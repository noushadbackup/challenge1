"""Choose the time-NMS gap on TRAINING OOF only (bundle oof_final.csv = replayed train OOF), per target.

    python3 work/I1_clip/nms_oof.py [bag]
"""
import json
import sys
from pathlib import Path

import pandas as pd

import uwbfall
from uwbfall import clip as CL
from uwbfall.scoring import gt_records, score_inproc, summary
from uwbfall.sources import events_for
from srcv import bridge, core as C

HERE = Path(__file__).resolve().parent
bag = sys.argv[1] if len(sys.argv) > 1 else "uwb_default"
GAPS = [-1, 0, 1, 2, 3, 4, 5]
res = {}
for t in ["0", "1", "2", "3", "official"]:
    tdir = "official_train" if t == "official" else f"fold_{t}"
    if t == "official":
        tr, ev = uwbfall.split_sources("train"), C.real_events("train")
    else:
        tr, ev = bridge.pipeline_sources(int(t), "train"), C.events(int(t), "train")
    gt = gt_records(events_for(sorted(tr, key=lambda s: s.sid), ev), ignore_flagged=True)
    dur = {s.sid: s.duration for s in tr}
    fin = pd.read_csv(C.DATA / "runs" / bag / tdir / "oof_final.csv")
    res[t] = {g: summary(score_inproc(CL.nms_time(fin, g), gt, dur))["rank_f1"] for g in GAPS}
    print(t, {g: round(v, 4) for g, v in res[t].items()}, flush=True)
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / f"nms_oof_{bag}.json").write_text(json.dumps(res, indent=1))

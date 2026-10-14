"""End-to-end fit() + predict() through the W3 clip stage (HGB-only to keep it cheap; the TCN path is unchanged).

Checks that fit() trains the junction threshold + segment classifier, gates the OOF before tuning the operating
point, applies the final NMS to the replayed OOF, writes clip.pkl, and that predict() on val runs with the gate
and NMS (rows validated by the official validate_predictions via scoring.validate_df), then scores val with the
official CLI.

    python3 uwbfall/tests/smoke_clip_fit.py [out_dir]
"""
import json
import sys
from pathlib import Path

import uwbfall
from uwbfall.scoring import official_cli, summary, validate_df
from srcv import core as C

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/data_raw/I1/smoke/fit_hgb_clip")
cfg = {"tcn": {"enabled": False}, "hgb": {"enabled": True},
       "clip": {"segclf": {"enabled": True}, "gate": "mult", "gate_floor": 0.5},
       "decode": {"one_event_per_clip": "time", "nms_gap_s": 4.0}}
b = uwbfall.fit(uwbfall.split_sources("train"), C.real_events("train"), cfg, out)
meta = json.loads((b / "meta.json").read_text())
print("fit:", json.dumps({k: meta[k] for k in ("postproc", "oof_rank_f1_tuned", "oof")}), flush=True)
assert (b / "clip.pkl").exists()
val = uwbfall.split_sources("val")
pred = uwbfall.predict(b, val)
dur = {s.sid: s.duration for s in val}
csv = out / "val.csv"
validate_df(pred, dur, csv)
rep = official_cli(csv, "val", out / "val.cli.json")
print("val (official CLI):", json.dumps(summary(rep) if "by_theta" in rep else rep), flush=True)

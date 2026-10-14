"""Parity of the promoted W3 configuration inside uwbfall against the I1 experiment code (work/I1_clip).

For each target, `pipeline.retune` re-runs the post-model stages of an SR-CV bag-0 bundle (HGB + TCN reused) with
the clip stage on (segment classifier, gate mult floor 0.7) and the final time NMS (4 s), then `pipeline.predict`
runs on the held-out streams. Checks:
  * the operating point tuned by fit's code path equals the experiment's (cache/ops_uwb_default_<t>7.json, "G")
  * the predictions equal work/I1_clip/preds/uwb_default/G7N4/<name>.csv (experiment: variants.py G --gate-floor
    0.7, then apply_nms.py time:4)

    python3 -m uwbfall.tests.parity_clip_retune [targets]      (targets: official,0  default both)
"""
import json
import sys
from pathlib import Path

import pandas as pd

import uwbfall
from uwbfall import pipeline as PL
from srcv import bridge, core as C

REPO = Path(__file__).resolve().parents[2]
I1 = REPO / "work" / "I1_clip"
RUNS = C.DATA / "runs" / "uwb_default"
TMP = Path("/home/user/data_raw/I1/retune")
OVER = {"boundary": {"gate": 0.3},
        "clip": {"segclf": {"enabled": True}, "gate": "mult", "gate_floor": 0.7},
        "decode": {"one_event_per_clip": "time", "nms_gap_s": 4.0}}

ok = True
targets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["official", "0"]
for t in targets:
    if t == "official":
        tr, ev, te, name, bdir = (uwbfall.split_sources("train"), C.real_events("train"), uwbfall.split_sources("val"),
                                  "val", RUNS / "official_train")
    else:
        k = int(t)
        tr, ev, te, name, bdir = (bridge.pipeline_sources(k, "train"), C.events(k, "train"),
                                  bridge.pipeline_sources(k, "test"), f"fold_{k}", RUNS / f"fold_{k}")
    out = PL.retune(bdir, tr, ev, OVER, TMP / f"G7N4_{t}")
    meta = json.loads((out / "meta.json").read_text())
    ref_op = json.loads((I1 / "cache" / f"ops_uwb_default_{t}7.json").read_text())["G"]["op"]
    same_op = list(meta["postproc"]) == list(ref_op)
    pred = PL.predict(out, te)
    pred.to_csv(out / f"{name}.csv", index=False)
    a = pd.read_csv(out / f"{name}.csv").to_csv(index=False)
    b = pd.read_csv(I1 / "preds" / "uwb_default" / "G7N4" / f"{name}.csv").to_csv(index=False)
    ok &= same_op and a == b
    print(f"[{t}] op {meta['postproc']} == experiment {ref_op}: {same_op}; {len(pred)} rows identical to "
          f"G7N4/{name}.csv: {a == b}; train OOF final {json.dumps(meta['oof'].get('final'))}", flush=True)
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

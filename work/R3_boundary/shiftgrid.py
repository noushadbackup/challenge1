"""Ablation: is the detector blend just a constant shift? Training-OOF grid of constant start/end offsets
(boundary.ds/de applied to replayed rows).   python3 shiftgrid.py"""
from common import *  # noqa
import json
from select_b import final_rows
from uwbfall.pipeline import load_bundle
from uwbfall.scoring import gt_records, score_inproc, summary
res = {}
for t in TARGETS:
    B = load_bundle(bundle_dir(t)); cfg = B["cfg"]
    D = pd.read_csv(DUMP / tname(t) / "D_train.csv")
    dur = {s.sid: s.duration for s in train_sources(t)}
    gt = gt_records(train_events(t), ignore_flagged=True)
    rows = []
    for ds in (0.0, -0.1, -0.2, -0.3):
        for de in (0.0, 0.1, -0.1):
            b = {**cfg["boundary"], "ds": ds, "de": de}
            sm = summary(score_inproc(final_rows(D, b, None, {}, dur, cfg), gt, dur))
            rows.append(dict(ds=ds, de=de, rank=sm["rank_f1"], f=sm["f1@0.1/0.3/0.5/0.7"]))
    R = pd.DataFrame(rows); res[tname(t)] = rows
    print(tname(t), R.sort_values("rank", ascending=False).head(3).to_string(index=False), flush=True)
(HERE / "shiftgrid.json").write_text(json.dumps(res))

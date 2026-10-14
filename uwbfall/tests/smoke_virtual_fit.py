"""Smoke test: fit + predict entirely on VIRTUAL sources (no h5), tiny config.
Training: val_04 rebuilt from the published composition + p1001_04 regenerated from a new plan (E1 seed 1001).
Labels come from the virtual writer's own events. Prints the in-process official score on the training streams.
Usage: PYTHONPATH=/home/user/challenge1 python3 -m uwbfall.tests.smoke_virtual_fit <out_dir>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/challenge1/work/E1_pseudo_test")
import e1lib  # noqa: E402
from abc2027.streams import finish_plan, prepare_split  # noqa: E402

from uwbfall import fit, predict  # noqa: E402
from uwbfall.paths import PKG  # noqa: E402
from uwbfall.scoring import gt_records, score_inproc, summary  # noqa: E402
from uwbfall.sources import VirtualSource, sources_from_composition  # noqa: E402

g = e1lib.val_groups()
plan = finish_plan(prepare_split("test", g["falls"], g["adl"], g["others"], rng=np.random.default_rng(1001)), 57.1864)
comp = pd.read_csv(PKG / "streams_composition.csv", dtype={"source_t0": float, "source_t1": float})
ev = pd.read_csv(PKG / "streams_events.csv", dtype=str)
srcs = sources_from_composition(comp, ev, ["val_04"]) + [VirtualSource("p1001_04", plan.streams[4].segments)]
cfg = {"cv": {"n_folds": 2}, "tcn": {"seeds": 1, "epochs": 2}, "hgb": {"max_iter": 30}}
out = fit(srcs, None, cfg, sys.argv[1])
pred = predict(out, srcs)
events = pd.concat([s.events() for s in srcs], ignore_index=True)
r = score_inproc(pred, gt_records(events), {s.sid: s.duration for s in srcs})
print("rows", len(pred), "in-sample score (smoke only):", summary(r))

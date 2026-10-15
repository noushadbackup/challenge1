"""Parity of the W5 hit model inside uwbfall (I3) against the experiment (work/I3_score).

Attach the pooled-bag SR-CV hit model with the fixed F1*/2 threshold (work/I3_score/fit_hit.py --fixed-tau 0.46)
to the bag-0 reference bundle of the official train streams (I1 parity bundle G7N4_official), run
pipeline.predict on val and compare with work/I3_score/preds/uwb_default/XHTPF/val.csv (rows sorted by stream,
start). Also: with score.hit_model off the same bundle still predicts the reference G7N4 val CSV.

    python3 -m uwbfall.tests.parity_hit
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

import uwbfall
from uwbfall import pipeline as PL

REPO = Path(__file__).resolve().parents[2]
I3 = REPO / "work" / "I3_score"
BUNDLE = Path("/home/user/data_raw/I1/retune/G7N4_official")
HIT = Path("/home/user/data_raw/I3/hit_XHTPF.pkl")
OUT = Path("/home/user/data_raw/I3/G7N4_hit_official")

te = uwbfall.split_sources("val")
ok = True
ref = PL.predict(BUNDLE, te)
a = ref.to_csv(index=False)
b = pd.read_csv(REPO / "work" / "I1_clip" / "preds" / "uwb_default" / "G7N4" / "val.csv").to_csv(index=False)
print(f"hit model off: identical to G7N4/val.csv: {a == b}", flush=True)
ok &= a == b
state = pickle.load(open(HIT, "rb"))
out = PL.attach_hit(BUNDLE, state, OUT)
pred = PL.predict(out, te)
key = ["stream_id", "t_start"]
x = pd.read_csv(__import__("io").StringIO(pred.to_csv(index=False))).sort_values(key).reset_index(drop=True)
y = pd.read_csv(I3 / "preds" / "uwb_default" / "XHTPF" / "val.csv").sort_values(key).reset_index(drop=True)
same = x.equals(y)
print(f"hit model on (tau {state['tau']}): {len(x)} rows, identical to XHTPF/val.csv: {same}", flush=True)
if not same:
    print(x.compare(y) if len(x) == len(y) else (len(x), len(y)))
ok &= same
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

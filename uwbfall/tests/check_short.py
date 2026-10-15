"""Regression check for short remainder streams (reports/round2/FIXES.md X1-X2).

    python3 -m uwbfall.tests.check_short [<bundle>]        (default: the repo copy models/final)

The last stream of a split holds the remaining segments and can be seconds long (the shortest ADL piece is 0.52 s).
Before X1 a stream under 2 s crashed `predict` and so the whole 5-stream submission. This builds single-stream
folders named `test` from crops of val_00 (from 20 s; a fall starts 8.34 s into the crop) in a temp dir. It
predicts each with the feature cache off and checks that:
  * nothing raises;
  * the official validate_predictions accepts the CSV;
  * crops under pipeline.MIN_PREDICT_S get no rows;
  * the 12 s crop finds the fall.
Exit 0 = OK.
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from abc2027 import evaluate as E

from uwbfall.paths import PKG
from uwbfall.pipeline import MIN_PREDICT_S, load_bundle, predict
from uwbfall.scoring import validate_df
from uwbfall.sources import dir_sources

REPO = Path(__file__).resolve().parents[2]
bundle = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "models" / "final")
B = load_bundle(bundle)
B["cfg"]["runtime"] = {**B["cfg"]["runtime"], "use_cache": False}
src = PKG / "streams" / "val" / "stream_00.h5"
bad = []
with tempfile.TemporaryDirectory() as td, h5py.File(src, "r") as h:
    for L in (0.52, 1.0, 1.9, 4.9, 6.0, 12.0):
        d = Path(td) / f"L{L:g}" / "test"
        d.mkdir(parents=True)
        i0, n = 20 * 50, int(round(L * 50))
        with h5py.File(d / "stream_04.h5", "w") as o:
            o["cir"] = np.asarray(h["cir"][:, i0:i0 + n], np.float32)
        srcs = dir_sources(d, "test")
        try:
            pred = predict(B, srcs, log=lambda *_: None)
            recs = validate_df(pred, E.stream_durations(d), d / "pred.csv")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{L} s: {type(e).__name__}: {e}")
            print(f"  {L:5.2f} s: FAILED {type(e).__name__}: {e}")
            continue
        ok = (len(recs) == 0) if L < MIN_PREDICT_S else (L < 10 or any(r["t_start"] < 9.12 and r["t_end"] > 8.34 for r in recs))
        print(f"  {L:5.2f} s: {len(recs)} rows {'OK' if ok else 'UNEXPECTED'}")
        if not ok:
            bad.append(f"{L} s: {len(recs)} rows")
print("SHORT-STREAM CHECK:", "OK" if not bad else "FAILED " + " | ".join(bad))
sys.exit(1 if bad else 0)

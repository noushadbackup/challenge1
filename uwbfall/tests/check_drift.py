"""Drift monitors react to synthetic drift and stay quiet on clean data.

    python3 -m uwbfall.tests.check_drift <bundle with drift_ref.json> [split=val]

Uses the cached features of the split's streams (no model run). Cases:
  clean          : the features as they are                                   -> expect no ALERT on the PSI/level rows
  scale x10      : CIR amplitude x10 (energies x100; a writer/unit change)    -> expect level + PSI ALERTs
  no repeats     : sample-and-hold repeats removed (e_diff zeros filled)      -> expect repeat-share + rep_frac ALERTs
  no junctions   : junction detector blind (writer without segment cuts)      -> expect junction-rate ALERT
"""
import sys

import numpy as np
import pandas as pd

from uwbfall import drift, split_sources
from uwbfall.extract import get_features
from uwbfall.pipeline import load_bundle

B = load_bundle(sys.argv[1])
split = sys.argv[2] if len(sys.argv) > 2 else "val"
ref = B["drift_ref"]
assert ref is not None, "bundle has no drift_ref.json (python -m uwbfall driftref ...)"
thr = ref["junction_thr"]
oof_like = pd.DataFrame(columns=["stream_id", "t_start", "t_end", "score"])


def run(name, mod, thr_=thr):
    stats = {}
    for s in split_sources(split):
        F = mod(dict(get_features(s, B["cfg"])))
        stats[s.sid] = drift.stream_stats(F, thr_)
    rep = drift.report(ref, stats, oof_like)
    alerts = sorted(c["monitor"] for c in rep["checks"] if c["flag"] == "ALERT" and c["monitor"] not in
                    ("events_per_h",))   # no predictions are passed here, so events/h is trivially 0
    print(f"{name:13s} ALERTs (excluding events/h): {alerts}")
    return set(alerts)


def scale(F):
    return {**F, "e_raw": F["e_raw"] * 100, "e_mot": F["e_mot"] * 100, "e_diff": F["e_diff"] * 100,
            "prof_02": F["prof_02"]}


def norep(F):
    e = F["e_diff"].copy()
    e[e == 0] = np.median(e[e > 0])
    return {**F, "e_diff": e}


a0 = run("clean", lambda F: F)
a1 = run("scale x10", scale)
a2 = run("no repeats", norep)
a3 = run("no junctions", lambda F: F, thr_=1e9)
ok = (not a0 and {"raw energy level r0", "raw energy level r1", "raw energy level r2"} <= a1
      and "repeat_share" in a2 and "junctions_per_h" in a3)
print("drift monitor check:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)

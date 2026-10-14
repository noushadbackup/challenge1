"""Train-OOF-only selection of the gate form (no test data): tuned detector OOF ranking F1 per gate mode.

    python3 work/I1_clip/gate_oof.py <bag> [targets]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uwbfall import clip as CL
from uwbfall.config import load_config
from uwbfall.decode import tune
from uwbfall.scoring import gt_records
from uwbfall.sources import events_for

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import RUNS, target_sources  # noqa: E402

HERE = Path(__file__).resolve().parent
bag = sys.argv[1]
targets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2", "3", "official"]
MODES = [("none", 0), ("mult", 0), ("sqrt", 0), ("mult", 0.3), ("mult", 0.5), ("mult", 0.7)]
res = {}
for t in targets:
    tdir = "official_train" if t == "official" else f"fold_{t}"
    cfg = load_config(json.loads((RUNS / "uwb_default" / tdir / "config.json").read_text()))
    tr, ev, _te, _ = target_sources(t)
    tr = sorted(tr, key=lambda s: s.sid)
    gt = gt_records(events_for(tr, ev), ignore_flagged=True)
    dur = {s.sid: s.duration for s in tr}
    o0, o1 = np.load(RUNS / "uwb_default" / tdir / "oof.npz"), np.load(RUNS / bag / tdir / "oof.npz")
    sid = o0["sid"]
    p0 = 0.75 * o1["oof_tcn"] + 0.25 * o0["oof_hgb"]
    G = pickle.load(open(HERE / "cache" / f"prep_{t}.pkl", "rb"))["gate"]
    g = np.concatenate([G[x] for x in pd.unique(sid)])
    res[t] = {}
    for mode, fl in MODES:
        p = CL.gate_probs(p0, g, mode, fl)
        sc, op, _ = tune({s: p[sid == s] for s in pd.unique(sid)}, gt, dur, cfg)
        res[t][f"{mode}_{fl}"] = round(sc, 4)
    print(t, res[t], flush=True)
df = pd.DataFrame(res)
df["mean"] = df.mean(1)
print(df.round(4))
(HERE / "out" / f"gate_oof_{bag}.json").write_text(json.dumps(res, indent=1))

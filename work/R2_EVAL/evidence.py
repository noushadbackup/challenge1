"""Per-event evidence for the replay x TCN puzzle on val (bag uwb_default): the events whose tIoU@0.3 status
flips under replay for C4 (TCN) or C3 (blend), with every crop's replay, the B median, the J median and the GT.
    python3 work/R2_EVAL/evidence.py [target] -> out/evidence_<target>.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import refine_variant, CROPS, JCROPS
from mech import gt_for
from uwbfall.boundary import tiou

HERE = Path(__file__).resolve().parent
t = sys.argv[1] if len(sys.argv) > 1 else "official"
g = gt_for(t); g = g[~g.ign]
rows = []
for c in ["C3", "C4"]:
    D = pd.read_csv(HERE / "cache" / f"D_uwb_default_{t}_{c}.csv")
    DJ = pd.read_csv(HERE / "cache" / f"DJ_uwb_default_{t}_{c}.csv")
    DD = pd.concat([D, DJ[[x for x in DJ.columns if any(x.startswith(j + "_") for j in JCROPS)]]], axis=1)
    sB, eB = refine_variant(D, "R"); sJ, eJ = refine_variant(DD, "J")
    for i, r in D.iterrows():
        gs = g[g.stream_id == r.stream_id]
        if not len(gs):
            continue
        di = tiou(r.t_start, r.t_end, gs.t_start.values, gs.t_end.values)
        j = int(np.argmax(di))
        if di[j] <= 0:
            continue
        iB = float(tiou(sB[i], eB[i], gs.t_start.values[j], gs.t_end.values[j]))
        iJ = float(tiou(sJ[i], eJ[i], gs.t_start.values[j], gs.t_end.values[j]))
        if (di[j] >= .3) != (iB >= .3) or (iB >= .3) != (iJ >= .3):
            crops = "; ".join(f"{k}:{DD[k + '_s'][i]:.2f}-{DD[k + '_e'][i]:.2f}"
                              + (f"[{DD[k + '_rev'][i]}]" if isinstance(DD[k + '_rev'][i], str) else "")
                              for k in CROPS + JCROPS if np.isfinite(DD[k + '_s'][i]))
            rows.append(dict(system=c, stream=r.stream_id, det=f"{r.t_start:.2f}-{r.t_end:.2f}",
                             gt=f"{gs.t_start.values[j]:.2f}-{gs.t_end.values[j]:.2f}", iou_det=round(float(di[j]), 2),
                             B=f"{sB[i]:.2f}-{eB[i]:.2f}", iou_B=round(iB, 2), J=f"{sJ[i]:.2f}-{eJ[i]:.2f}",
                             iou_J=round(iJ, 2), crops=crops))
X = pd.DataFrame(rows)
X.to_csv(HERE / "out" / f"evidence_{t}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 400)
print(X.drop(columns="crops").to_string())
for r in X.itertuples():
    print(r.system, r.stream, r.det, "->", r.crops)

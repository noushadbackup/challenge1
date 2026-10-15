"""Resume-check: re-score fresh-stream preds with official evaluator; check fresh compositions differ from originals
and stay subject/file-disjoint from fold train."""
import sys, os, tempfile
from pathlib import Path
import pandas as pd
from abc2027 import evaluate as E
F = Path("/home/user/data_raw/review_fresh"); R = Path("/home/user/data_raw/srcv")
gt, dur = [], {}
for k in range(4):
    gt += E.load_ground_truth(F / f"fold_{k}/test_events.csv", ignore_flagged=True)
    c = pd.read_csv(F / f"fold_{k}/test_composition.csv")
    for sid, g in c.groupby("stream_id"): dur[sid] = float(g.t_end.max())
def recs(files):
    df = pd.concat([pd.read_csv(f) for f in files]); fd, p = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    df.to_csv(p, index=False); r = E.validate_predictions(Path(p), dur); os.unlink(p); return r
print("fresh evaluable", sum(not g["ignore"] for g in gt), "streams", len(dur), "h", round(sum(dur.values())/3600, 3))
for s in ("C3", "C3G3", "N4", "FINAL"):
    fs = sorted((F / "preds" / s).glob("fold_*.csv"))
    r = E.evaluate(gt, recs(fs), dur); b = r["by_theta"]["0.3"]
    print(s, len(fs), round(r["ranking_score"]["f1_mean"], 4), (b["tp"], b["fp"], b["fn"]))
# composition comparison + disjointness
print(open(R/"fold_0/test_composition.csv").readline().strip())
for k in range(4):
    a = pd.read_csv(R / f"fold_{k}/test_composition.csv"); b = pd.read_csv(F / f"fold_{k}/test_composition.csv")
    tr = pd.read_csv(R / f"fold_{k}/train_composition.csv") if (R / f"fold_{k}/train_composition.csv").exists() else None
    pcol = [c for c in a.columns if "path" in c or "file" in c or "source" in c][0]
    key = lambda d: set(zip(d.stream_id, d[pcol], d.t_start.round(2)))
    same = len(key(a) & key(b)) / len(b)
    msg = f"fold {k}: rows orig {len(a)} fresh {len(b)}; identical (stream,file,t_start) {same:.3f}; same file set {set(a[pcol])==set(b[pcol])}"
    if tr is not None:
        msg += f"; fresh∩train files {len(set(b[pcol]) & set(tr[pcol]))}"
        if "subject" in b.columns: msg += f"; fresh∩train subjects {len(set(b.subject) & set(tr.subject))}"
    print(msg)

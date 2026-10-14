"""Paired segment-cluster bootstrap of val ranking F1 differences (official matching code).

Every TP / FN / FP at theta 0.1/0.3/0.5 is computed with abc2027.evaluate._match_stream, then attributed to
the composition segment (junction-to-junction piece) containing its midpoint; segments are resampled.
  python3 bootstrap.py ref_csv cand_csv [cand_csv ...]
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import E, OUT, PKG, durations, gt  # noqa: E402

TH = (0.1, 0.3, 0.5)
co = pd.read_csv(PKG / "streams_composition.csv")
co = co[co.stream_id.str.startswith("val")].reset_index(drop=True)
seg_of = lambda sid, t: int(co.index[(co.stream_id == sid) & (co.t_start <= t) & (co.t_end > t)][0])  # noqa: E731


def counts(csv):
    d = durations("val")
    preds = E.validate_predictions(Path(csv), d)
    G = gt()
    C = np.zeros((len(co), len(TH), 3))  # tp, fp, fn per segment
    for j, th in enumerate(TH):
        for sid in d:
            g = [e for e in G if e["stream_id"] == sid]
            p = [x for x in preds if x["stream_id"] == sid]
            r = E._match_stream(g, p, th)
            matched = {pi for pi, _, _ in r["pairs"]}
            hit = {gi for _, gi, _ in r["pairs"]}
            ign = [e for e in g if e["ignore"]]
            for pi, x in enumerate(p):
                mid = 0.5 * (x["t_start"] + x["t_end"])
                if pi in matched:
                    C[seg_of(sid, mid), j, 0] += 1
                elif not any(E._intervals_overlap(x["t_start"], x["t_end"], e["t_start"] - E.IGNORE_MARGIN_S, e["t_end"] + E.IGNORE_MARGIN_S) for e in ign):
                    C[seg_of(sid, mid), j, 1] += 1
            for gi, e in enumerate(r["scored"]):
                if gi not in hit:
                    C[seg_of(sid, 0.5 * (e["t_start"] + e["t_end"])), j, 2] += 1
    return C


def rank(C):
    tp, fp, fn = C[..., 0].sum(0), C[..., 1].sum(0), C[..., 2].sum(0)
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))


ref = sys.argv[1]
Cr = counts(ref)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(co), (5000, len(co)))
res = {"ref": Path(ref).name, "ref_rank_f1_mean_theta_0.1_0.3_0.5": round(rank(Cr), 4), "n_segments": len(co), "cands": {}}
for cand in sys.argv[2:]:
    Cc = counts(cand)
    dd = np.array([rank(Cc[i]) - rank(Cr[i]) for i in idx])
    res["cands"][Path(cand).name] = {"rank_f1": round(rank(Cc), 4), "delta": round(rank(Cc) - rank(Cr), 4),
                                     "ci95": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)],
                                     "p_delta_gt_0": round(float((dd > 0).mean()), 3)}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / f"bootstrap_vs_{Path(ref).stem}.json", "w"), indent=1)

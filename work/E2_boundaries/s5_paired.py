"""Paired comparison prototype vs refined on one split: stream-cluster bootstrap and an
event-level bootstrap (GT events resampled, FP counts held fixed -> optimistic). Official matcher.
Usage: python3 s5_paired.py <split> <a.csv> <b.csv>"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from s3_select import GT, DUR, _match_stream, per_stream_counts, boot_delta

split, A, B = sys.argv[1:4]
pa, pb = (pd.read_csv(p).to_dict("records") for p in (A, B))
print("stream-cluster bootstrap (mean, 90% CI, P>0):", boot_delta(per_stream_counts(pa, split), per_stream_counts(pb, split)))
def hits(P):
    out = {}
    for th in (0.1, 0.3, 0.5):
        h, fp = [], 0
        for sid in sorted(DUR[split]):
            G = [g for g in GT if g["stream_id"] == sid]; Q = [p for p in P if p["stream_id"] == sid]
            r = _match_stream(G, Q, th); m = {gi for _, gi, _ in r["pairs"]}
            h += [i in m for i in range(len(r["scored"]))]; fp += r["false_positive"]
        out[th] = (np.array(h), fp)
    return out
a, b = hits(pa), hits(pb)
for th in (0.1, 0.3, 0.5):
    print(f"theta {th}: gained {int((~a[th][0] & b[th][0]).sum())} lost {int((a[th][0] & ~b[th][0]).sum())} FP {a[th][1]}->{b[th][1]}")
rng = np.random.default_rng(0); n = len(a[0.1][0]); d = []
for _ in range(5000):
    i = rng.integers(0, n, n); s = 0
    for th in (0.1, 0.3, 0.5):
        for (h, fp), sg in ((a[th], -1), (b[th], 1)):
            tp = h[i].sum(); s += sg * 2 * tp / (2 * tp + fp + n - tp) / 3
    d.append(s)
print("event bootstrap delta mean %.4f 95%% CI [%.4f, %.4f]" % (np.mean(d), *np.percentile(d, [2.5, 97.5])))

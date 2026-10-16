"""Diagnostic on TRAINING streams only (labels/composition of training streams are training data):
does the annotator replayed on the TRUE fall clip (junction to junction) reproduce the GT?  If yes, the
boundary problem reduces to locating the fall clip's junctions precisely.

    python3 oracle_clip.py <target> [max_streams]
"""
from common import *  # noqa
from srcv import core as C
from abc2027.annotate import propose_fall
from uwbfall.boundary import tracks_for
from uwbfall.paths import RATE_HZ as FS

t = "val" if sys.argv[1] == "val" else int(sys.argv[1])
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 99
comp = C.composition(t, "train")
ev = unflagged(C.events(t, "train"))
rows = []
for s in sorted(train_sources(t), key=lambda s: s.sid)[:nmax]:
    f = comp[(comp.stream_id == s.sid) & (comp.kind == "fall")]
    e = ev[ev.stream_id == s.sid]
    for r in e.itertuples():
        g0, g1 = float(r.t_start), float(r.t_end)
        clip = f[(f.t_start.astype(float) <= g0) & (f.t_end.astype(float) >= g1)]
        if not len(clip):
            continue
        a0, a1 = float(clip.t_start.iloc[0]), float(clip.t_end.iloc[0])
        out = dict(sid=s.sid, gs=g0, ge=g1, clip0=a0, clip1=a1)
        for nm, d0, d1 in (("c0", 0.0, 0.0), ("c2", 0.2, 0.2), ("c4", 0.4, 0.4), ("c2s", 0.2, 3.0)):
            ia, ib = int(round((a0 + d0) * FS)), int(round((a1 - d1) * FS))
            cir = np.asarray(s.crop(ia, ib))
            p = propose_fall(tracks_for(cir, True), (ib - ia) / FS)
            ps, pe = ia / FS + p.t_start, ia / FS + p.t_end
            out[nm] = float(tiou(ps, pe, g0, g1))
            out[nm + "_es"] = ps - g0
            out[nm + "_ee"] = pe - g1
        rows.append(out)
    s.release()
P = pd.DataFrame(rows)
P.to_csv(HERE / f"oracle_clip_{tname(t)}.csv", index=False)
for nm in ("c0", "c2", "c4", "c2s"):
    print(nm, "n", len(P), "hit@.5 %.3f @.7 %.3f @.9 %.3f" % tuple((P[nm] >= th).mean() for th in (0.5, 0.7, 0.9)),
          "med|es| %.3f med|ee| %.3f" % (P[nm + "_es"].abs().median(), P[nm + "_ee"].abs().median()))

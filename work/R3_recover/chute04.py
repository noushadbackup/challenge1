"""(c) Chute04 error analysis on TRAINING OOF of every base bundle (no test data used for anything but counting).

For every training stream: the junction segments, their segment-classifier features (the 13 'phys' features the
norep bundle uses), the subject-grouped OOF P(fall clip) (recovered from the gated OOF), the true fall clip the
segment overlaps and the fall type. Prints how Chute04 fall segments differ from other fall segments, separately
for well-segmented clips (both junctions found within 0.6 s) and mis-segmented ones.

    python3 work/R3_recover/chute04.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import srcv
from srcv import core as C
import uwbfall
from uwbfall import clip as CL
from uwbfall import pipeline as P
from uwbfall.extract import get_features
from uwbfall.features import design, stream_blocks

import rules as RL

FEATS = ['len_s', 'j_left', 'j_right', 'static_ab_max', 'st5_max', 'st10_max', 'st10_p90', 'diff_max', 'diff_p90',
         'diff_mean']


def table(target: str) -> pd.DataFrame:
    d = RL.load(target, "train")
    B = P.load_bundle(f"/home/user/data_raw/srcv/runs/strict/norep/{target}")
    cfg = B["cfg"]
    srcs = uwbfall.split_sources("train") if target == "official_train" else srcv.pipeline_sources(int(target[-1]), "train")
    ev = d["events"].copy()
    ev["t_start"] = ev.t_start.astype(float); ev["t_end"] = ev.t_end.astype(float)
    out = []
    for s in srcs:
        F = get_features(s, cfg)
        blk = stream_blocks(F, cfg)
        xp, _xb, cols = design(blk, B["pca"], cfg)
        fpc = [i for i, c in enumerate(cols) if c.startswith("fp") and not c.startswith("fpseg")]
        segs, tab, _pk, _st = CL.stream_segments(F, blk["base"], xp[:, fpc], B["clip"]["thr"], cfg)
        S = d["streams"][s.sid]
        assert (segs == S["segs"]).all()
        comp = CL.composition_of(s)
        tab["y"] = CL.segment_labels(segs, comp)
        tab["gate"] = S["gate"][segs[:, 0]]
        tab["pmax"] = [S["p"][a:b].max() for a, b in segs]
        tab["rawmax"] = [S["raw"][a:b].max() for a, b in segs]
        tab["sid"], tab["a"], tab["b"] = s.sid, segs[:, 0] * RL.STEP, segs[:, 1] * RL.STEP
        ft, segok = [], []
        E = ev[ev.stream_id == s.sid]
        fc = comp[comp.kind == "fall"]
        for a, b in segs * RL.STEP:
            ov = np.minimum(fc.t_end.values, b) - np.maximum(fc.t_start.values, a)
            if len(fc) and ov.max() > 0.5 * (b - a):
                c = fc.iloc[int(np.argmax(ov))]
                e = E[(E.t_start >= c.t_start - 1e-6) & (E.t_end <= c.t_end + 1e-6)]
                ft.append(e.fall_type.iloc[0] if len(e) else "?")
                segok.append(abs(a - c.t_start) <= 0.6 and abs(b - c.t_end) <= 0.6)
            else:
                ft.append(""); segok.append(False)
        tab["fall_type"], tab["seg_ok"], tab["target"] = ft, segok, target
        out.append(tab)
        s.release()
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    T = pd.concat([table(t) for t in ["fold_0", "fold_1", "fold_2", "fold_3", "official_train"]])
    keep = [c for c in T.columns if not c.startswith("fpm") and not c.startswith("rep_")]
    T[keep].round(4).to_csv(RL.Path(__file__).resolve().parent / "diag" / "chute04_segments.csv.gz", index=False)
    f = T[T.fall_type.str.startswith("Chute")]
    pd.set_option("display.width", 250)
    print("fall segments by type: n, share well-segmented, gate median, share gate<0.3 (well-seg / mis-seg)")
    for ft, g in f.groupby("fall_type"):
        ok, bad = g[g.seg_ok], g[~g.seg_ok]
        print(f"  {ft}: n={len(g)} seg_ok={g.seg_ok.mean():.2f}  gate_med ok={ok.gate.median():.3f} bad={bad.gate.median():.3f}"
              f"  low-gate ok={(ok.gate < .3).mean():.3f} bad={(bad.gate < .3).mean():.3f}")
    ok = f[f.seg_ok]
    print("\nwell-segmented fall segments: feature medians by type (and Chute04 split by gate)")
    rows = {ft: g[FEATS].median() for ft, g in ok.groupby("fall_type")}
    c4 = ok[ok.fall_type == "Chute04"]
    rows["C04 gate<.3"] = c4[c4.gate < .3][FEATS].median()
    rows["C04 gate>=.3"] = c4[c4.gate >= .3][FEATS].median()
    rows["ADL segs"] = T[T.y == 0][FEATS].median()
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("\nraw (ungated) blend max in fall segments, Chute04 low-gate vs ADL segments with gate<.3:")
    print("  C04 low-gate rawmax quantiles", np.round(np.quantile(c4[c4.gate < .3].rawmax, [.1, .25, .5, .75, .9]), 3))
    adl = T[(T.y == 0) & (T.gate < .3)]
    print("  ADL low-gate rawmax quantiles", np.round(np.quantile(adl.rawmax, [.5, .9, .95, .99, .999]), 3), "n", len(adl))

"""Boundary refinement by annotator replay (E2 config B).

The ground truth is produced by the package's abc2027.annotate.propose_fall. For each candidate event we
run it on short raw-CIR crops around the detection, on distinct frames (bit-exact sample-and-hold repeats
removed, true frame times kept), take the median start/end over the crops whose replay agrees with the
detector (tIoU >= gate), and apply a 1.0 s duration floor by extending the end only.

Port of work/E2_boundaries/s2_replay.py (only the crop variants named in the config; refine_junction is
not needed for them) and s3_select.refine / to_preds.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, median_filter

from .paths import RATE_HZ, SERIALS
from .sources import StreamSource
from abc2027.annotate import RadarTrack, propose_fall  # noqa: E402

FS = RATE_HZ
_CROP = re.compile(r"^(loc|detq|det)([0-9.]+)_([0-9.]+)(d?)$")


def parse_crop(name: str):
    """'loc2_3d' -> (use_junction, L, R, dedup); det -> q97 junctions, detq -> q95 (recall) junctions."""
    m = _CROP.match(name)
    if not m:
        raise ValueError(f"unsupported crop {name!r} (refined-junction 'detr*' variants are not ported)")
    kind, L, R, d = m.groups()
    uj = {"loc": False, "det": True, "detq": "q95"}[kind]
    return uj, float(L), float(R), d == "d"


def detected_junctions(prof_02: np.ndarray, q: float = 97):
    """Variance-normalised static-profile jump on 0.2 s blocks (X2/F4 definition): times (s), strength."""
    p = prof_02.astype(np.float32) / 1e3
    z = p[..., :184] + 1j * p[..., 184:]
    norm = np.median(np.linalg.norm(z, axis=2), axis=1, keepdims=True) + 1e-12
    d = np.zeros(z.shape[:2])
    d[:, 1:-1] = np.linalg.norm(z[:, 2:] - z[:, :-2], axis=2)
    j = (d / norm).max(0)
    jn = j / (median_filter(j, size=51, mode="nearest") + 1e-9)
    thr = np.percentile(jn, q)
    pk = np.flatnonzero((jn > thr) & (jn == maximum_filter(jn, size=11)))
    return (pk + 0.5) * 0.2, jn[pk]


def tracks_for(cir, dedup):
    tr = []
    for i, s in enumerate(SERIALS):
        x = cir[i].astype(np.float64)
        t = np.arange(len(x)) / FS
        if dedup:
            keep = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
            x, t = x[keep], t[keep]
        d = np.diff(x, axis=0)
        tr.append(RadarTrack(serial=s, time_s=t, cir=x, motion=np.einsum("ij,ij->i", d, d)))
    return tr


def replay_stream(source: StreamSource, prof_02: np.ndarray, cands: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw replay bounds per crop for the candidates of one stream (same columns as E2 replay_*.csv)."""
    b = cfg["boundary"]
    crops = [(c, *parse_crop(c)) for c in b["crops"]]
    Tn = source.n_frames
    T = Tn / FS
    J, _ = detected_junctions(prof_02, b["junction_q"])
    J95, _ = detected_junctions(prof_02, b["junction_q_recall"])
    rows = []
    for r in cands.itertuples():
        g = 1.0
        prev = J[J < r.t_start - g]; nxt = J[J > r.t_end + g]
        jp = prev[-1] if len(prev) else None
        jn = nxt[0] if len(nxt) else None
        prev = J95[J95 < r.t_start - 0.2]; nxt = J95[J95 > r.t_end + 0.2]
        jp2 = prev[-1] if len(prev) else None
        jn2 = nxt[0] if len(nxt) else None
        lo = max(0.0, min(r.t_start - 10.5, (jp if jp is not None else r.t_start) - 1.0,
                          (jp2 if jp2 is not None else r.t_start) - 1.0))
        hi = min(T, max(r.t_end + 5.5, (jn if jn is not None else r.t_end) + 1.0))
        i_lo, i_hi = int(lo * FS), int(np.ceil(hi * FS))
        cir = np.asarray(source.crop(i_lo, i_hi))
        t0 = i_lo / FS
        out = dict(stream_id=r.stream_id, t_start=r.t_start, t_end=r.t_end, score=r.score,
                   jprev=jp if jp is not None else np.nan, jnext=jn if jn is not None else np.nan,
                   jprev2=jp2 if jp2 is not None else np.nan)
        for name, uj, L, R, dd in crops:
            a, bb = r.t_start - L, r.t_end + R
            if uj:
                pj, nj = (jp2, jn2) if uj == "q95" else (jp, jn)
                if pj is not None:
                    a = max(a, pj + 0.2)
                if nj is not None:
                    bb = min(bb, nj - 0.2)
            a, bb = max(a, lo, 0.0), min(bb, hi, T)
            ia, ib = int(round((a - t0) * FS)), int(round((bb - t0) * FS))
            if ib - ia < 150:  # < 3 s crop: not usable
                out[f"{name}_s"] = out[f"{name}_e"] = np.nan
                continue
            a = t0 + ia / FS
            p = propose_fall(tracks_for(cir[:, ia:ib, :], dd), (ib - ia) / FS)
            out[f"{name}_s"], out[f"{name}_e"] = a + p.t_start, a + p.t_end
            out[f"{name}_pk"] = a + p.t_peak
            out[f"{name}_rev"] = ";".join(p.reasons)
            out[f"{name}_lowq"] = p.quality == "low"
            out[f"{name}_anchor"] = SERIALS.index(p.anchor_serial)
            out[f"{name}_spread"] = p.onset_spread_s
        rows.append(out)
    return pd.DataFrame(rows)


def tiou(a0, a1, b0, b1):
    i = np.maximum(0, np.minimum(a1, b1) - np.maximum(a0, b0))
    u = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(u > 0, i / np.where(u > 0, u, 1), 0.0)


def refine(D: pd.DataFrame, b: dict):
    """Port of E2 s3_select.refine: refined (t_start, t_end) arrays and the replaced mask."""
    s0, e0 = D.t_start.values.astype(float), D.t_end.values.astype(float)
    S = np.vstack([D[f"{c}_s"].values for c in b["crops"]]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in b["crops"]]).astype(float)
    ok = np.isfinite(S) & np.isfinite(E) & (E > S) & (tiou(S, E, s0, e0) >= b["gate"])
    if b.get("fallback", "detector") == "contain":
        # I2: a candidate whose crops all fail the tIoU gate (typically a detector interval much wider than the
        # fall) uses the crops lying mostly inside the detector interval instead of the raw detector bounds
        inter = np.maximum(0, np.minimum(E, e0) - np.maximum(S, s0))
        inside = np.isfinite(S) & np.isfinite(E) & (E > S) & (inter >= b.get("contain_frac", 0.8) * (E - S))
        none = ok.sum(0) == 0
        ok = ok | (inside & none[None, :])
    if b.get("agg", "median") == "cluster":
        ok = ok & _largest_cluster(S, E, ok, b.get("cluster_iou", 0.8))
    Sm = np.where(ok, S, np.nan); Em = np.where(ok, E, np.nan)
    use = ok.sum(0) >= b.get("min_ok", 1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rs, re_ = np.nanmedian(Sm, 0), np.nanmedian(Em, 0)
    s = np.where(use, rs + b.get("ds", 0.0), s0 + b.get("pds", 0.0))
    e = np.where(use, re_ + b.get("de", 0.0), e0 + b.get("pde", 0.0))
    e = np.maximum(e, s + b.get("dmin", 0.0))
    return s, e, use


def _largest_cluster(S, E, ok, thr):
    """Mode-aware crop aggregation (I2, R2_EVAL open issue): among the agreeing crops of a candidate, keep the
    largest group of crops whose replay has tIoU >= thr with a medoid crop (the medoid maximises the group size;
    ties -> larger summed tIoU, then the earlier crop). A 3-vs-3 split of two bursts then no longer averages the
    two modes; with a single mode it equals the plain median."""
    keep = np.zeros_like(ok)
    for i in range(S.shape[1]):
        idx = np.flatnonzero(ok[:, i])
        if len(idx) <= 1:
            keep[idx, i] = True
            continue
        T = tiou(S[idx, i][:, None], E[idx, i][:, None], S[idx, i][None, :], E[idx, i][None, :])
        A = T >= thr
        key = [(A[j].sum(), T[j][A[j]].sum(), -j) for j in range(len(idx))]
        j = max(range(len(idx)), key=lambda q: key[q])
        keep[idx[A[j]], i] = True
    return keep


def to_preds(D: pd.DataFrame, s, e, durations: dict[str, float]) -> pd.DataFrame:
    out = []
    for sid, a, bb, sc in zip(D.stream_id, s, e, D.score):
        a = max(0.0, a); bb = min(durations[sid], bb)
        if bb > a:
            out.append({"stream_id": sid, "t_start": round(float(a), 3), "t_end": round(float(bb), 3),
                        "score": float(sc), "t_alarm": round(float(a), 3)})
    return pd.DataFrame(out, columns=["stream_id", "t_start", "t_end", "score", "t_alarm"])


# ----------------------------------------------------------------------------- W4 hypothesis selector (I2)
# Each candidate gets boundary hypotheses: the reference median of agreeing crops ("med", if any crop agrees),
# every single crop replay, and the detector bounds ("det"); all with the duration floor. A LightGBM regressor
# predicts each hypothesis's tIoU with the ground truth from replay internals (crop agreement, spread, anchor
# radar, annotator quality flags, offsets to the detector and to junctions); the argmax replaces the median when
# its predicted tIoU exceeds the median's by more than `margin`. Every hypothesis is an annotator output or the
# detector, so no padding/shift is learned (start-error tie-break protected).

SEL_FEATURES = ["is_med", "is_det", "crop_i", "t_det", "t_med", "support8", "mean_t_crops", "dur", "ds", "de",
                "det_dur", "score", "n_ok", "spread_s", "spread_e", "lowq", "n_rev", "anchor", "onset_spread",
                "to_jprev", "to_jnext", "use"]


def hypotheses(D: pd.DataFrame, b: dict) -> pd.DataFrame:
    """Long table (one row per candidate x hypothesis) with SEL_FEATURES and columns row (candidate index), hs, he."""
    s_med, e_med, use = refine(D, b)
    s0, e0 = D.t_start.values.astype(float), D.t_end.values.astype(float)
    crops = b["crops"]
    dmin = b.get("dmin", 0.0)
    S = np.vstack([D[f"{c}_s"].values for c in crops]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in crops]).astype(float)
    fin = np.isfinite(S) & np.isfinite(E) & (E > S)
    E2 = np.where(fin, np.maximum(E, S + dmin), np.nan)
    ok = fin & (tiou(S, E, s0, e0) >= b["gate"])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sp_s, sp_e = np.nanstd(np.where(fin, S, np.nan), 0), np.nanstd(np.where(fin, E, np.nan), 0)
    jp = D.jprev.values.astype(float) if "jprev" in D else np.full(len(D), np.nan)
    jn = D.jnext.values.astype(float) if "jnext" in D else np.full(len(D), np.nan)
    rows = []
    for i in range(len(D)):
        H = []
        if use[i]:
            H.append((s_med[i], e_med[i], 1, 0, -1))
        for c in range(len(crops)):
            if fin[c, i]:
                H.append((S[c, i], E2[c, i], 0, 0, c))
        H.append((s0[i], max(e0[i], s0[i] + dmin), 0, 1, -1))
        hs = np.array([h[0] for h in H]); he = np.array([h[1] for h in H])
        cs, ce = S[fin[:, i], i], E2[fin[:, i], i]
        for (a, z, im, idt, ci) in H:
            tc = tiou(cs, ce, a, z) if len(cs) else np.zeros(0)
            nm = crops[ci] if ci >= 0 else None
            rev = D[f"{nm}_rev"].iloc[i] if nm and f"{nm}_rev" in D else ""
            rows.append(dict(row=i, hs=a, he=z, is_med=im, is_det=idt, crop_i=ci,
                             t_det=float(tiou(a, z, s0[i], e0[i])),
                             t_med=float(tiou(a, z, s_med[i], e_med[i])) if use[i] else np.nan,
                             support8=int((tc >= 0.8).sum()), mean_t_crops=float(tc.mean()) if len(tc) else np.nan,
                             dur=z - a, ds=a - s0[i], de=z - e0[i], det_dur=e0[i] - s0[i],
                             score=float(D.score.iloc[i]) if "score" in D else np.nan, n_ok=int(ok[:, i].sum()),
                             spread_s=sp_s[i], spread_e=sp_e[i],
                             lowq=float(D[f"{nm}_lowq"].iloc[i]) if nm and f"{nm}_lowq" in D else np.nan,
                             n_rev=float(len([x for x in str(rev).split(";") if x and x != "nan"])) if nm else np.nan,
                             anchor=float(D[f"{nm}_anchor"].iloc[i]) if nm and f"{nm}_anchor" in D else np.nan,
                             onset_spread=float(D[f"{nm}_spread"].iloc[i]) if nm and f"{nm}_spread" in D else np.nan,
                             to_jprev=a - jp[i], to_jnext=jn[i] - z, use=int(use[i])))
    return pd.DataFrame(rows)


def selector_targets(H: pd.DataFrame, D: pd.DataFrame, gt: pd.DataFrame, min_iou: float = 0.1) -> np.ndarray:
    """tIoU of each hypothesis with the candidate's ground-truth event (the evaluable GT event of the stream with
    the largest tIoU over the candidate's hypotheses); NaN for candidates without a GT event >= min_iou."""
    y = np.full(len(H), np.nan)
    for i, g in H.groupby("row"):
        sid = D.stream_id.iloc[i]
        ev = gt[gt.stream_id == sid]
        if not len(ev):
            continue
        T = np.stack([tiou(g.hs.values, g.he.values, a, z) for a, z in zip(ev.t_start.values, ev.t_end.values)])
        j = int(np.argmax(T.max(1)))
        if T[j].max() >= min_iou:
            y[g.index.values] = T[j]
    return y


def fit_selector(H: pd.DataFrame, y: np.ndarray, seed: int = 0):
    import lightgbm as lgb
    m = np.isfinite(y)
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                             deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1).fit(H.loc[m, SEL_FEATURES], y[m])


def select(D: pd.DataFrame, b: dict, model, margin: float = 0.0, H: pd.DataFrame | None = None):
    """Selector boundaries: (t_start, t_end, replaced-mask) like refine(). Falls back to refine() where the best
    predicted hypothesis does not beat the reference hypothesis (med if used, else det) by more than margin."""
    s, e, use = refine(D, b)
    if H is None:
        H = hypotheses(D, b)
    if not len(H):
        return s, e, use
    H = H.assign(pred=model.predict(H[SEL_FEATURES]))
    s, e = s.copy(), e.copy()
    for i, g in H.groupby("row"):
        ref = g[(g.is_med == 1)] if use[i] else g[g.is_det == 1]
        j = g.pred.idxmax()
        if g.pred[j] > ref.pred.iloc[0] + margin:
            s[i], e[i] = g.hs[j], g.he[j]
    return s, e, use

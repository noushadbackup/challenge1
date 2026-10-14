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


def to_preds(D: pd.DataFrame, s, e, durations: dict[str, float]) -> pd.DataFrame:
    out = []
    for sid, a, bb, sc in zip(D.stream_id, s, e, D.score):
        a = max(0.0, a); bb = min(durations[sid], bb)
        if bb > a:
            out.append({"stream_id": sid, "t_start": round(float(a), 3), "t_end": round(float(bb), 3),
                        "score": float(sc), "t_alarm": round(float(a), 3)})
    return pd.DataFrame(out, columns=["stream_id", "t_start", "t_end", "score", "t_alarm"])

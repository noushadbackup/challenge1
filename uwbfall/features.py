"""0.2 s block features: the prototype's 64 "full" features + E4's 48 range-band channels (= 112).

Ported verbatim from analysis/prototype_detector.py (block_features, structure_features, add_fingerprint,
cols_for) and work/E4_deep_model/data.py (band_features). The only change is the input: a dict of extractor
arrays instead of a cached npz path, and per-stream processing (no concatenation until model fitting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .paths import STEP_S

STEP = STEP_S


def shift(x: np.ndarray, k: int) -> np.ndarray:
    """x[t+k] with edge padding, along the last axis."""
    if k == 0:
        return x
    y = np.roll(x, -k, axis=-1)
    if k > 0:
        y[..., -k:] = x[..., -1:]
    else:
        y[..., :-k] = x[..., :1]
    return y


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 0)]), axis=-1)
    return (c[..., k:] - c[..., :-k]) / k


def block_features(F: dict) -> tuple[dict, np.ndarray]:
    p = F["prof_02"].astype(np.float32) / 1e3
    z = p[..., :184] + 1j * p[..., 184:]  # (3, n, 184)
    n = z.shape[1]
    norm = np.median(np.linalg.norm(z, axis=2), axis=1, keepdims=True) + 1e-12
    f: dict[str, np.ndarray] = {}
    jump = np.linalg.norm(shift(z.transpose(0, 2, 1), 1) - shift(z.transpose(0, 2, 1), -1), axis=1) / norm
    f["jump_max"] = np.log10(jump.max(0) + 1e-6)
    for k in (5, 10):
        zz = z.transpose(0, 2, 1)
        m = rolling_mean(zz, k)
        past = np.concatenate([np.repeat(m[..., :1], k, axis=-1), m], axis=-1)[..., :n]
        fut = np.concatenate([m[..., 1:], np.repeat(m[..., -1:], k, axis=-1)], axis=-1)[..., :n]
        ch = np.log10(np.linalg.norm(fut - past, axis=1) / norm + 1e-6)
        for r in range(3):
            f[f"static{k}_r{r}"] = ch[r]
        f[f"static{k}_max"] = ch.max(0)
    ed = F["e_diff"].astype(np.float64)[:, : n * 10].reshape(3, n, 10).mean(2)
    ed = np.log10(ed / np.median(ed, axis=1, keepdims=True) + 1e-6)
    em = F["e_mot"].astype(np.float64)[:, : n * 10].reshape(3, n, 10).mean(2)
    em = np.log10(em / np.median(em, axis=1, keepdims=True) + 1e-6)
    for r in range(3):
        f[f"diff_r{r}"] = ed[r]
        f[f"ema_r{r}"] = em[r]
    f["diff_max"] = ed.max(0)
    for key in ("static5_max", "static10_max", "diff_max"):
        for k in (-10, -5, 5, 10):
            f[f"{key}@{k * STEP:+.0f}s"] = shift(f[key], k)
    amp = np.log10(np.abs(z).transpose(1, 0, 2).reshape(n, -1) + 1e-9)
    return f, amp


def structure_features(f: dict):
    j = f["jump_max"]
    thr = np.percentile(j, 97)
    peaks = np.flatnonzero((j > thr) & (j >= shift(j, 1)) & (j >= shift(j, -1)))
    n = len(j)
    idx = np.arange(n)
    if len(peaks) == 0:
        peaks = np.array([0, n - 1])
    pos = np.searchsorted(peaks, idx, side="right")
    prev = np.where(pos > 0, peaks[np.clip(pos - 1, 0, len(peaks) - 1)], 0)
    nxt = np.where(pos < len(peaks), peaks[np.clip(pos, 0, len(peaks) - 1)], n - 1)
    out = {
        "t_since_jump": np.minimum((idx - prev) * STEP, 60),
        "t_until_jump": np.minimum((nxt - idx) * STEP, 60),
    }
    out["seg_len"] = np.minimum(out["t_since_jump"] + out["t_until_jump"], 120)
    out["seg_frac"] = out["t_since_jump"] / np.maximum(out["seg_len"], STEP)
    return out, prev, nxt


def band_features(F: dict, n: int, bands) -> tuple[np.ndarray, list[str]]:
    mm = F["mot_map"][:, :n].astype(np.float32)
    p = F["prof_02"][:, :n].astype(np.float32) / 1e3
    zc = p[..., :184] + 1j * p[..., 184:]
    dz = np.abs(np.roll(zc, -1, axis=1) - np.roll(zc, 1, axis=1))
    dz[:, 0], dz[:, -1] = dz[:, 1], dz[:, -2]
    cols, names = [], []
    for nm, arr in (("mot", mm), ("dprof", dz)):
        for r in range(3):
            for a, b in bands:
                v = np.log10(arr[r, :, a:b].mean(1) + 1e-9)
                cols.append(v - np.median(v))
                names.append(f"{nm}_r{r}_b{a}")
    return np.stack(cols, 1).astype(np.float32), names


def stream_blocks(F: dict, cfg: dict) -> dict:
    """Everything a stream contributes before the train-fitted fingerprint: base DataFrame (prototype columns
    in prototype order, plus _prev/_next), log-amplitude matrix for the PCA, band channels."""
    f, amp = block_features(F)
    s, prev, nxt = structure_features(f)
    n = len(f["jump_max"])
    d = pd.DataFrame({**f, **s})
    d["_prev"], d["_next"] = prev, nxt
    bands, bnames = band_features(F, n, cfg["features"]["bands"])
    out = {"base": d, "amp": amp, "bands": bands, "band_names": bnames, "n": n,
           "t": (np.arange(n) + 0.5) * STEP}
    if cfg["features"].get("zones"):
        out["zones"], out["zone_names"] = zone_features(F, n, cfg["features"]["zones"])
    return out


def zone_features(F: dict, n: int, zones) -> tuple[np.ndarray, list[str]]:
    """Per-radar range-zone features (PLAN v2 W4 D3 hedge; data verdict F5/D-b: D3 falls are seen by R29171 at
    far bins ~40-68, D1/D2 by R29444/R30722 at bins ~5-21). The whole-profile prototype features dilute a far-bin
    change with the strong near-bin clutter, so each zone [radar, a, b) gets its own:
      zs5/zs10 static change (past vs future k-block complex-profile mean, normalised by the zone's median norm),
      zmot motion (mot_map zone mean), zjump frame-to-frame profile jump in the zone (log10, stream-median removed),
    plus the max over zones of zs10 shifted by -2/-1/+1/+2 s. Per-stream only (no cross-stream information)."""
    p = F["prof_02"][:, :n].astype(np.float32) / 1e3
    z = p[..., :184] + 1j * p[..., 184:]  # (3, n, 184)
    mm = F["mot_map"][:, :n].astype(np.float32)
    cols, names, s10 = [], [], []
    for r, a, b in zones:
        zz = z[r][:, a:b].T  # (bins, n)
        norm = np.median(np.linalg.norm(zz, axis=0)) + 1e-12
        tag = f"z{r}_{a}"
        for k in (5, 10):
            m = rolling_mean(zz, k)
            past = np.concatenate([np.repeat(m[..., :1], k, axis=-1), m], axis=-1)[..., :n]
            fut = np.concatenate([m[..., 1:], np.repeat(m[..., -1:], k, axis=-1)], axis=-1)[..., :n]
            v = np.log10(np.linalg.norm(fut - past, axis=0) / norm + 1e-6)
            cols.append(v); names.append(f"zs{k}_{tag}")
            if k == 10:
                s10.append(v)
        v = np.log10(mm[r, :, a:b].mean(1) + 1e-9)
        cols.append(v - np.median(v)); names.append(f"zmot_{tag}")
        j = np.linalg.norm(shift(zz, 1) - shift(zz, -1), axis=0) / norm
        v = np.log10(j + 1e-6)
        cols.append(v - np.median(v)); names.append(f"zjump_{tag}")
    smax = np.max(s10, axis=0)
    for k in (-10, -5, 5, 10):
        cols.append(shift(smax, k)); names.append(f"zs10max@{k * STEP:+.0f}s")
    return np.stack(cols, 1).astype(np.float64), names


def hgb_design(blk: dict, xp: np.ndarray, cfg: dict) -> np.ndarray:
    """HGB member input: the prototype design, plus the zone features when features.zones is set and
    features.zones_for includes 'hgb'. The TCN input (prototype + bands) is not changed by the zones."""
    if cfg["features"].get("zones") and "hgb" in cfg["features"]["zones_for"]:
        return np.hstack([xp, blk["zones"]])
    return xp


def fit_pca(amps: list[np.ndarray], cfg: dict) -> PCA:
    a = np.vstack(amps)
    return PCA(cfg["features"]["pca_components"], random_state=0).fit(a[:: cfg["features"]["pca_stride"]])


def proto_frame(blk: dict, pca: PCA) -> pd.DataFrame:
    d = blk["base"].copy()
    pc = pca.transform(blk["amp"])
    seg_id = d["_prev"].values
    pooled = pd.DataFrame(pc).groupby(seg_id).transform("mean").values
    for i in range(pc.shape[1]):
        d[f"fp{i}"] = pc[:, i]
        d[f"fpseg{i}"] = pooled[:, i]
    return d


def cols_for(df: pd.DataFrame, fs: str) -> list[str]:
    base = [c for c in df.columns if c.startswith(("static", "diff", "jump"))]
    if fs == "physics":
        return base
    struct = base + [c for c in df.columns if c.startswith(("ema", "t_since", "t_until", "seg_"))]
    if fs == "structure":
        return struct
    return struct + [c for c in df.columns if c.startswith("fp")]


def design(blk: dict, pca: PCA, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X_proto float64 (n, 64), X_band float32 (n, 48), proto column names)."""
    d = proto_frame(blk, pca)
    cols = cols_for(d, cfg["features"]["proto_set"])
    return d[cols].values.astype(np.float64), blk["bands"], cols


def ignore_clip_mask(events: pd.DataFrame, comp: pd.DataFrame, sid: str, n: int, onset_s: float = 0.25,
                     hard_cut_s: float = 0.05) -> np.ndarray:
    """Block-centre mask of whole fall clips that are generator ignore zones (data verdict N5): the clip holds a
    flagged fall (quality low or needs_review) whose onset is within `onset_s` of the clip start (junction-start
    fall), or the clip starts with a hard cut (overlap with the previous segment < hard_cut_s, or stream start).
    Unflagged (evaluable-type) falls are never masked, so their clip keeps its labels."""
    t = (np.arange(n) + 0.5) * STEP
    m = np.zeros(n, bool)
    c = comp.sort_values("t_start").reset_index(drop=True)
    for i in np.flatnonzero((c.kind == "fall").values):
        a, b = float(c.t_start[i]), float(c.t_end[i])
        ov = float(c.t_end[i - 1]) - a if i > 0 else 0.0
        ev = events[(events.stream_id == sid) & (events.t_start.astype(float) >= a - 1e-6)
                    & (events.t_end.astype(float) <= b + 1e-6)]
        for r in ev.itertuples():
            fl = str(r.quality) == "low" or str(r.needs_review).lower() == "true"
            if fl and (float(r.t_start) - a < onset_s or ov < hard_cut_s):
                m |= (t >= a) & (t < b)
    return m


def block_labels(events: pd.DataFrame, sid: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-centre labels as analysis/common.frame_labels: fall and flagged masks."""
    t = (np.arange(n) + 0.5) * STEP
    fall = np.zeros(n, bool)
    flagged = np.zeros(n, bool)
    ev = events[events.stream_id == sid]
    for row in ev.itertuples():
        m = (t >= float(row.t_start)) & (t < float(row.t_end))
        fl = str(row.quality) == "low" or str(row.needs_review).lower() == "true"
        fall |= m
        flagged |= m & fl
    return fall, flagged

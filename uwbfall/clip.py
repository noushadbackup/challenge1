"""Clip-level stage (PLAN v2 W3; experts X2 P0-2/P0-3/P1-1 and X3 P0-3/P0-4).

* Junction detector: variance-normalised two-window statistic on the 0.2 s complex static profile of all 3 radars
  (data-team verdict D-a / N8, L2's definition): s(i) = |m_after - m_before|^2 / (v_before + v_after), windows of W
  blocks. Peaks above a threshold chosen on the TRAINING streams for recall of fall-clip junctions (default 0.95).
* Segment classifier: LightGBM multiclass (adl / fall clip / autre-empty) on junction-to-junction segments.
  Features: static change across the segment, motion burst, PCA fingerprint (segment mean), sample-and-hold repeat
  shares (spacing 2/3/4, within the stream only), segment length, junction strengths.
* Decoding helpers: hard 1-D NMS by time gap, or at most one event per junction-to-junction segment.
* Post-fall class: block labels fall / post-fall-in-clip / other for a 3-class GBDT.

Everything is within one stream; nothing uses other streams at prediction time (no transductive cue).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .paths import PKG, STEP_S

STEP = STEP_S
KINDS = ("adl", "fall", "other")          # segment classes (index = label)
SEG_FEATS_VERSION = "clip-v1"


# ----------------------------------------------------------------------------- composition
_PKG_COMP = None


def composition_of(source) -> pd.DataFrame:
    """Composition rows (t_start, t_end, kind, label) of a source: the in-memory plan of a VirtualSource, or the
    package's streams_composition.csv for a published h5 stream (training labels only)."""
    global _PKG_COMP
    if hasattr(source, "composition"):
        c = source.composition()
    else:
        if _PKG_COMP is None:
            _PKG_COMP = pd.read_csv(PKG / "streams_composition.csv")
        c = _PKG_COMP[_PKG_COMP.stream_id == source.sid]
    cols = ["t_start", "t_end", "kind", "label"] + (["path"] if "path" in c.columns else [])
    c = c[cols].copy()
    c["t_start"] = c.t_start.astype(float)
    c["t_end"] = c.t_end.astype(float)
    return c.sort_values("t_start").reset_index(drop=True)


# ----------------------------------------------------------------------------- junctions
def junction_stat(prof_02: np.ndarray, W: int = 5) -> np.ndarray:
    """log10 variance-normalised two-window jump at each block boundary i (blocks [i-W,i) vs [i,i+W))."""
    p = prof_02.astype(np.float64) / 1e3
    z = (p[..., :184] + 1j * p[..., 184:]).transpose(1, 0, 2).reshape(p.shape[1], -1)  # (n, 3*184)
    n = len(z)
    cs = np.concatenate([np.zeros((1, z.shape[1]), complex), np.cumsum(z, 0)])
    cs2 = np.concatenate([[0.0], np.cumsum((np.abs(z) ** 2).sum(1))])
    s = np.full(n, -9.0)
    i = np.arange(W, n - W + 1)
    if len(i):
        mb = (cs[i] - cs[i - W]) / W
        ma = (cs[i + W] - cs[i]) / W
        vb = (cs2[i] - cs2[i - W]) / W - (np.abs(mb) ** 2).sum(1)
        va = (cs2[i + W] - cs2[i]) / W - (np.abs(ma) ** 2).sum(1)
        d = (np.abs(ma - mb) ** 2).sum(1)
        s[i[i < n]] = np.log10(d / (np.maximum(vb, 0) + np.maximum(va, 0) + 1e-9) + 1e-9)[i < n]
    return s


def junction_peaks(stat: np.ndarray, thr: float, min_sep_s: float = 1.0) -> np.ndarray:
    pk, _ = find_peaks(stat, height=thr, distance=max(1, int(round(min_sep_s / STEP))))
    return pk


def true_junctions(comp: pd.DataFrame) -> pd.DataFrame:
    """Junction times (start of every row but the first) and whether they bound a fall clip."""
    t = comp.t_start.values[1:]
    kind_after = comp.kind.values[1:]
    kind_before = comp.kind.values[:-1]
    return pd.DataFrame({"t": t, "fall": (kind_after == "fall") | (kind_before == "fall"),
                         "into_fall": kind_after == "fall"})


def junction_threshold(stats: dict, comps: dict, recall: float = 0.95, tol_blocks: int = 2) -> float:
    """Threshold on the statistic giving `recall` on fall-clip junctions of the training streams
    (per-junction max within +-tol blocks)."""
    vals = []
    for sid, s in stats.items():
        J = true_junctions(comps[sid])
        for t in J.t.values[J.fall.values]:
            i = int(round(t / STEP))
            vals.append(s[max(i - tol_blocks, 0): i + tol_blocks + 1].max())
    return float(np.quantile(vals, 1 - recall))


def junction_recall(stats: dict, comps: dict, thr: float, min_sep_s: float, tol_s: float = 0.6) -> dict:
    """Diagnostics: recall of fall / all junctions and precision of detected peaks (+-tol_s)."""
    tp_f = n_f = tp_a = n_a = n_pk = pk_ok = 0
    for sid, s in stats.items():
        J = true_junctions(comps[sid])
        pk = junction_peaks(s, thr, min_sep_s) * STEP
        n_pk += len(pk)
        for t, f in zip(J.t.values, J.fall.values):
            hit = len(pk) and np.min(np.abs(pk - t)) <= tol_s
            n_a += 1; tp_a += bool(hit)
            if f:
                n_f += 1; tp_f += bool(hit)
        if len(J):
            pk_ok += int((np.min(np.abs(pk[:, None] - J.t.values[None]), 1) <= tol_s).sum()) if len(pk) else 0
    return {"recall_fall_junctions": tp_f / max(n_f, 1), "recall_all_junctions": tp_a / max(n_a, 1),
            "peak_precision": pk_ok / max(n_pk, 1), "peaks_per_min": n_pk / max(sum(len(s) for s in stats.values()) * STEP / 60, 1e-9)}


def segments_from_peaks(n: int, peaks: np.ndarray) -> np.ndarray:
    """(m, 2) block ranges [a, b) between consecutive junction peaks (stream ends included)."""
    b = np.unique(np.r_[0, peaks, n])
    return np.stack([b[:-1], b[1:]], 1)


# ----------------------------------------------------------------------------- segment features
def repeat_flags(e_diff: np.ndarray) -> np.ndarray:
    """Frame f repeats frame f-1 bit-exactly on all 3 radars (sample-and-hold), within this stream only."""
    return np.r_[False, np.all(e_diff[:, 1:] == 0, axis=0)]


def segment_table(F: dict, base: pd.DataFrame, fp: np.ndarray, stat: np.ndarray, segs: np.ndarray) -> pd.DataFrame:
    """One row per segment. base = stream_blocks(...)['base'] (prototype block features), fp = PCA scores (n, k)."""
    p = F["prof_02"].astype(np.float32) / 1e3
    z = p[..., :184] + 1j * p[..., 184:]                         # (3, n, 184)
    norm = np.median(np.linalg.norm(z, axis=2), axis=1) + 1e-12   # (3,)
    rep = repeat_flags(F["e_diff"])
    st5, st10, dm = base["static5_max"].values, base["static10_max"].values, base["diff_max"].values
    rows = []
    for a, b in segs:
        L = b - a
        e = max(1, min(5, L // 3))
        d0 = np.linalg.norm(z[:, a:a + e].mean(1) - z[:, b - e:b].mean(1), axis=1) / norm
        g = min(3, max(L // 4, 0))
        inner = slice(a + g, b - g) if b - g > a + g else slice(a, b)
        f0, f1 = a * 10, b * 10
        ri = np.flatnonzero(rep[f0:f1])
        sp = np.diff(ri)
        r = {"len_s": L * STEP, "j_left": stat[a] if a > 0 else 3.0, "j_right": stat[b] if b < len(stat) else 3.0,
             "static_ab_max": np.log10(d0.max() + 1e-6), "static_ab_r0": np.log10(d0[0] + 1e-6),
             "static_ab_r1": np.log10(d0[1] + 1e-6), "static_ab_r2": np.log10(d0[2] + 1e-6),
             "st5_max": st5[inner].max(), "st10_max": st10[inner].max(), "st10_p90": np.percentile(st10[inner], 90),
             "diff_max": dm[inner].max(), "diff_p90": np.percentile(dm[inner], 90), "diff_mean": dm[inner].mean(),
             "rep_rate": rep[f0:f1].mean() if f1 > f0 else 0.0,
             "rep_sp2": (sp == 2).mean() if len(sp) >= 5 else np.nan,
             "rep_sp3": (sp == 3).mean() if len(sp) >= 5 else np.nan,
             "rep_sp4": (sp == 4).mean() if len(sp) >= 5 else np.nan,
             "rep_n": len(ri)}
        fm = fp[a:b].mean(0)
        for i, v in enumerate(fm):
            r[f"fpm{i}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def segment_labels(segs: np.ndarray, comp: pd.DataFrame) -> np.ndarray:
    """Class (KINDS index) of the composition row with the largest overlap."""
    t0, t1 = segs[:, 0] * STEP, segs[:, 1] * STEP
    ov = np.minimum(comp.t_end.values[None], t1[:, None]) - np.maximum(comp.t_start.values[None], t0[:, None])
    k = comp.kind.values[np.argmax(ov, 1)]
    return np.array([KINDS.index(x) for x in k])


def subject_of_path(path: str) -> str:
    """Fall subject (Test_P{d}_{n}) for falls/ files, ADL_{n} for activities/<Act>_<n>.json (grouping only)."""
    import re
    path = str(path)
    if path.startswith("falls/"):
        return path.split("/")[2]
    m = re.search(r"_(\d+)\.json$", path)
    return f"ADL_{int(m.group(1))}" if m else path


def segment_subjects(segs: np.ndarray, comp: pd.DataFrame) -> np.ndarray:
    t0, t1 = segs[:, 0] * STEP, segs[:, 1] * STEP
    ov = np.minimum(comp.t_end.values[None], t1[:, None]) - np.maximum(comp.t_start.values[None], t0[:, None])
    return np.array([subject_of_path(x) for x in comp.path.values[np.argmax(ov, 1)]])


def feature_set(feats: list[str], name: str) -> list[str]:
    fp = [f for f in feats if f.startswith("fpm")]
    rep = [f for f in feats if f.startswith("rep_")]
    return {"all": feats, "nofp": [f for f in feats if f not in fp], "norep": [f for f in feats if f not in rep],
            "phys": [f for f in feats if f not in fp and f not in rep],
            "phys+sp3": [f for f in feats if f not in fp and (f not in rep or f == "rep_sp3")]}[name]


def stream_segments(F: dict, base: pd.DataFrame, fp: np.ndarray, thr: float, cfg: dict):
    """Junction statistic -> peaks -> segments -> segment feature table, for one stream."""
    c = cfg["clip"]
    stat = junction_stat(F["prof_02"], c["junction_W"])
    pk = junction_peaks(stat, thr, c["junction_min_sep_s"])
    segs = segments_from_peaks(len(stat), pk)
    return segs, segment_table(F, base, fp, stat, segs), pk * STEP, stat


def lgbm_segclf(cfg: dict):
    import lightgbm as lgb
    c = cfg["clip"]["segclf"]
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
                              num_leaves=c["num_leaves"], min_child_samples=c["min_child_samples"],
                              subsample=1.0, colsample_bytree=c["colsample_bytree"], reg_lambda=1.0,
                              random_state=0, deterministic=True, force_row_wise=True,
                              n_jobs=int(cfg["runtime"]["threads"]), verbose=-1)


def block_gate(segs: np.ndarray, p_fallclip: np.ndarray, n: int) -> np.ndarray:
    g = np.zeros(n)
    for (a, b), v in zip(segs, p_fallclip):
        g[a:b] = v
    return g


def gate_probs(p: np.ndarray, g: np.ndarray, mode: str, floor: float = 0.0) -> np.ndarray:
    if mode == "none":
        return p
    if mode == "mult":
        return p * (floor + (1 - floor) * g)
    if mode == "sqrt":
        return p * np.sqrt(floor + (1 - floor) * g)
    raise ValueError(mode)


# ----------------------------------------------------------------------------- decoding (hard 1-D NMS)
def nms_time(df: pd.DataFrame, gap: float) -> pd.DataFrame:
    """Greedy by score (desc): drop a row whose interval lies within `gap` s of an already kept row of the same
    stream (gap = max(0, distance between intervals)). gap <= 0 disables the rule except for overlap (< 0 -> off)."""
    if gap < 0 or not len(df):
        return df
    keep = []
    for sid, g in df.groupby("stream_id", sort=False):
        g = g.sort_values("score", ascending=False, kind="stable")
        kept = []
        for i, r in zip(g.index, g.itertuples()):
            if any(max(r.t_start - k1, k0 - r.t_end) < gap or (gap == 0 and min(r.t_end, k1) > max(r.t_start, k0))
                   for k0, k1 in kept):
                continue
            kept.append((r.t_start, r.t_end))
            keep.append(i)
    return df.loc[sorted(keep)]


def nms_segment(df: pd.DataFrame, junctions: dict, max_span: float = 18.7, guard: float = 0.4) -> pd.DataFrame:
    """At most one row per junction-to-junction segment (greedy by score). Junctions lying inside a row's interval
    (+-guard) are ignored (the fall's own static change is a jump). Two rows are in the same segment when no
    remaining junction separates them and their span is <= max_span (longest fall clip in the generator)."""
    if not len(df):
        return df
    keep = []
    for sid, g in df.groupby("stream_id", sort=False):
        J = np.asarray(junctions.get(sid, []), float)
        inside = np.zeros(len(J), bool)
        for r in g.itertuples():
            inside |= (J > r.t_start - guard) & (J < r.t_end + guard)
        J = J[~inside]
        g = g.sort_values("score", ascending=False, kind="stable")
        kept = []
        for i, r in zip(g.index, g.itertuples()):
            m = 0.5 * (r.t_start + r.t_end)
            clash = False
            for k0, k1 in kept:
                km = 0.5 * (k0 + k1)
                lo, hi = min(m, km), max(m, km)
                if max(r.t_end, k1) - min(r.t_start, k0) <= max_span and not ((J > lo) & (J < hi)).any():
                    clash = True
                    break
            if not clash:
                kept.append((r.t_start, r.t_end))
                keep.append(i)
    return df.loc[sorted(keep)]


# ----------------------------------------------------------------------------- post-fall class
def postfall_labels(events: pd.DataFrame, comp: pd.DataFrame, sid: str, n: int, after_s: float = 0.5) -> np.ndarray:
    """Block-centre mask of 'post-fall in clip': inside the fall clip holding an event, later than t_end + after_s."""
    t = (np.arange(n) + 0.5) * STEP
    m = np.zeros(n, bool)
    clips = comp[comp.kind == "fall"]
    for r in events[events.stream_id == sid].itertuples():
        a, b = float(r.t_start), float(r.t_end)
        c = clips[(clips.t_start <= a + 1e-6) & (clips.t_end >= b - 1e-6)]
        if not len(c):
            continue
        m |= (t >= b + after_s) & (t < float(c.t_end.iloc[0]))
    return m

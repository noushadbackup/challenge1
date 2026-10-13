"""Verification prototype: frame-level (0.2 s) gradient boosting + event post-processing.

Usage: python analysis/prototype_detector.py [feature_set ...]
  feature sets: physics  -> static change + frame-difference motion only
                structure -> + junction / segment-structure features
                full      -> + background fingerprint (PCA of static amplitude)

Post-processing is tuned on out-of-fold predictions of the TRAIN streams only
(leave-streams-out GroupKFold), then applied once to the validation streams and
scored with the official evaluator (--ignore-flagged).
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from common import FEATS, PKG, REPO, events, frame_labels, stream_ids

STEP = 0.2
OUT = REPO / "reports" / "prototype"
OUT.mkdir(parents=True, exist_ok=True)


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
    return (c[..., k:] - c[..., :-k]) / k  # length n-k+1, window [t, t+k)


def block_features(sid: str) -> tuple[dict, np.ndarray]:
    F = np.load(FEATS / f"{sid}.npz")
    p = F["prof_02"].astype(np.float32) / 1e3
    z = p[..., :184] + 1j * p[..., 184:]  # (3, n, 184)
    n = z.shape[1]
    norm = np.median(np.linalg.norm(z, axis=2), axis=1, keepdims=True) + 1e-12
    f: dict[str, np.ndarray] = {}
    # junction-sized jump between neighbouring blocks
    jump = np.linalg.norm(shift(z.transpose(0, 2, 1), 1) - shift(z.transpose(0, 2, 1), -1), axis=1) / norm
    f["jump_max"] = np.log10(jump.max(0) + 1e-6)
    # static change past-vs-future at two scales (fall = standing -> lying)
    for k in (5, 10):
        zz = z.transpose(0, 2, 1)
        m = rolling_mean(zz, k)  # mean over [t, t+k)
        past = np.concatenate([np.repeat(m[..., :1], k, axis=-1), m], axis=-1)[..., :n]  # mean [t-k, t)
        fut = np.concatenate([m[..., 1:], np.repeat(m[..., -1:], k, axis=-1)], axis=-1)[..., :n]  # mean [t+1, t+1+k)
        ch = np.log10(np.linalg.norm(fut - past, axis=1) / norm + 1e-6)
        for r in range(3):
            f[f"static{k}_r{r}"] = ch[r]
        f[f"static{k}_max"] = ch.max(0)
    # junction-immune motion: frame-difference energy, per radar, relative to stream median
    ed = F["e_diff"].astype(np.float64)[:, : n * 10].reshape(3, n, 10).mean(2)
    ed = np.log10(ed / np.median(ed, axis=1, keepdims=True) + 1e-6)
    em = F["e_mot"].astype(np.float64)[:, : n * 10].reshape(3, n, 10).mean(2)
    em = np.log10(em / np.median(em, axis=1, keepdims=True) + 1e-6)
    for r in range(3):
        f[f"diff_r{r}"] = ed[r]
        f[f"ema_r{r}"] = em[r]
    f["diff_max"] = ed.max(0)
    # temporal context of the key signals
    for key in ("static5_max", "static10_max", "diff_max"):
        for k in (-10, -5, 5, 10):
            f[f"{key}@{k * STEP:+.0f}s"] = shift(f[key], k)
    # amplitude for the fingerprint (log |z| per radar/bin), returned separately
    amp = np.log10(np.abs(z).transpose(1, 0, 2).reshape(n, -1) + 1e-9)
    return f, amp


def structure_features(f: dict) -> dict:
    """Time since / until the nearest strong jump, segment length."""
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


def build(sids: list[str]) -> pd.DataFrame:
    frames = []
    for sid in sids:
        f, amp = block_features(sid)
        s, prev, nxt = structure_features(f)
        n = len(f["jump_max"])
        lab = frame_labels(sid, n, STEP)
        d = pd.DataFrame({**f, **s})
        d["sid"], d["t"], d["y"] = sid, lab["t"], lab["fall"].astype(int)
        d["_prev"], d["_next"] = prev, nxt
        frames.append((d, amp))
    return frames


def add_fingerprint(frames, pca: PCA | None):
    amps = np.vstack([a for _d, a in frames])
    if pca is None:
        pca = PCA(16, random_state=0).fit(amps[::5])
    out = []
    for d, a in frames:
        pc = pca.transform(a)
        # pool the fingerprint over the enclosing jump-to-jump segment
        seg_id = d["_prev"].values
        pooled = pd.DataFrame(pc).groupby(seg_id).transform("mean").values
        for i in range(pc.shape[1]):
            d[f"fp{i}"] = pc[:, i]
            d[f"fpseg{i}"] = pooled[:, i]
        out.append(d)
    return pd.concat(out, ignore_index=True), pca


def cols_for(df: pd.DataFrame, fs: str) -> list[str]:
    base = [c for c in df.columns if c.startswith(("static", "diff", "jump"))]
    if fs == "physics":
        return base
    struct = base + [c for c in df.columns if c.startswith(("ema", "t_since", "t_until", "seg_"))]
    if fs == "structure":
        return struct
    return struct + [c for c in df.columns if c.startswith("fp")]


def to_events(sid: str, t: np.ndarray, p: np.ndarray, thr: float, smooth: int, min_d: float, gap: float):
    ps = np.convolve(p, np.ones(smooth) / smooth, "same") if smooth > 1 else p
    m = ps >= thr
    idx = np.flatnonzero(m)
    ev = []
    if len(idx) == 0:
        return ev
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts, stops = np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]]
    merged = []
    for a, b in zip(starts, stops):
        if merged and (a - merged[-1][1]) * STEP <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    for a, b in merged:
        if (b - a + 1) * STEP < min_d:
            continue
        ev.append((sid, a * STEP, (b + 1) * STEP, float(ps[a : b + 1].max()), a * STEP))
    return ev


def write_sub(evs, path):
    pd.DataFrame(evs, columns=["stream_id", "t_start", "t_end", "score", "t_alarm"]).round(3).to_csv(path, index=False)


def official(pred_path, split: str, out_json) -> dict:
    cmd = [sys.executable, "-m", "abc2027.evaluate", "--gt", str(PKG / "streams_events.csv"), "--pred", str(pred_path),
           "--streams", str(PKG / "streams" / split), "--ignore-flagged", "--out", str(out_json)]
    subprocess.run(cmd, check=True, cwd=PKG, env={"PYTHONPATH": str(PKG / "src"), "PATH": "/usr/bin:/bin"}, capture_output=True)
    return json.load(open(out_json))



def main() -> None:
    sets = sys.argv[1:] or ["physics", "structure", "full"]
    sids = stream_ids()
    tr_ids = [s for s in sids if s.startswith("train")]
    va_ids = [s for s in sids if s.startswith("val")]
    tr, pca = add_fingerprint(build(tr_ids), None)
    va, _ = add_fingerprint(build(va_ids), pca)
    results = {}
    grid = [(thr, sm, md, gp) for thr in (0.3, 0.4, 0.5, 0.6, 0.7) for sm in (1, 3, 5) for md in (0.4, 0.8) for gp in (0.2, 0.6)]
    for fs in sets:
        cols = cols_for(tr, fs)
        params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        oof = np.zeros(len(tr))
        for k, (a, b) in enumerate(GroupKFold(4).split(tr, tr.y, tr.sid)):
            m = HistGradientBoostingClassifier(**params).fit(tr.iloc[a][cols], tr.y.iloc[a])
            oof[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]
        best = None
        for g in grid:
            evs = [e for sid, d in tr.assign(p=oof).groupby("sid") for e in to_events(sid, d.t.values - STEP / 2, d.p.values, *g)]
            write_sub(evs, OUT / "tmp_train.csv")
            r = official(OUT / "tmp_train.csv", "train", OUT / "tmp_train.json")
            sc = r["ranking_score"]["f1_mean"]
            if best is None or sc > best[0]:
                best = (sc, g)
        model = HistGradientBoostingClassifier(**params).fit(tr[cols], tr.y)
        pv = model.predict_proba(va[cols])[:, 1]
        evs = [e for sid, d in va.assign(p=pv).groupby("sid") for e in to_events(sid, d.t.values - STEP / 2, d.p.values, *best[1])]
        write_sub(evs, OUT / f"val_{fs}.csv")
        r = official(OUT / f"val_{fs}.csv", "val", OUT / f"val_{fs}.json")
        bt = r["by_theta"]
        results[fs] = {
            "n_features": len(cols),
            "train_oof_rank_f1": round(best[0], 4),
            "postproc(thr,smooth,min_dur,gap)": best[1],
            "val_rank_f1": round(r["ranking_score"]["f1_mean"], 4),
            "val_f1@0.1/0.3/0.5/0.7": [round(bt[k]["f1"], 3) for k in ("0.1", "0.3", "0.5", "0.7")],
            "val_tp_fp_fn@0.3": [bt["0.3"]["tp"], bt["0.3"]["fp"], bt["0.3"]["fn"]],
            "val_fa_per_hour@0.3": round(bt["0.3"]["false_alarms_per_hour"], 2),
            "val_start_err_median@0.3": round(bt["0.3"]["start_error_median_abs"] or -1, 3),
            "val_mean_tiou@0.3": round(bt["0.3"]["mean_tiou"] or -1, 3),
        }
        print(fs, json.dumps(results[fs]), flush=True)
    json.dump(results, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

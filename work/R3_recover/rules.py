"""Recovery rules: add rows for falls the base system (final_norep) leaves without any row.

All rules work on one stream at a time with that stream's own gated blend probability p (0.2 s blocks), its
junction segments and their P(fall clip) (gate), and the base system's rows/candidates for that stream. Nothing is
compared across streams.

  sec   (idea b) a second, lower decode threshold applied only inside segments with P(fall clip) >= tau:
        to_events(p * [gate >= tau], thr2, smooth, min_d2, gap); a secondary event is kept only if it does not
        overlap a base detector candidate and lies more than `nms` s from every base row (the base 4 s NMS rule).
  amax  (idea a, probability version) for each segment with gate >= tau, no base row overlapping it and
        max p >= pmin: one row on the connected region around the arg-max where p >= frac * max p.
Added rows get score = scale * (their peak), with scale < 1 so that they rank below base rows of the same peak.

Boundaries: the detector interval ("det"), or the annotator replay on it (see replay.py), selected per variant.
"""
from __future__ import annotations

import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from uwbfall import clip as CL
from uwbfall.decode import events_df, to_events
from uwbfall.scoring import gt_records, score_inproc

R = Path("/home/user/data_raw/r3/recover")
STEP = 0.2
COLS = ["stream_id", "t_start", "t_end", "score", "t_alarm"]


def load(target: str, part: str) -> dict:
    with open(R / target / f"{part}.pkl", "rb") as f:
        d = pickle.load(f)
    if "postproc" not in d:
        import json
        meta = json.loads((Path("/home/user/data_raw/srcv/runs/strict/norep") / target / "meta.json").read_text())
        d["postproc"] = tuple(meta["postproc"])
    for S in d["streams"].values():
        if "raw" not in S:   # test part: ungated blend from the members (same expression as pipeline._blend)
            S["raw"] = 0.75 * S["tcn"] + 0.25 * S["hgb"]
    if part == "train":
        d["rows"], d["cands_base"] = d["oof_final"], d["oof_candidates"]
    else:
        d["rows"], d["cands_base"] = d["pred"], d["cands"]
    return d


def gt_train(d: dict) -> list[dict]:
    return gt_records(d["events"], ignore_flagged=True)


def _far(a, b, rows, gap):
    """True if [a, b] is more than `gap` s away from every row (same distance as clip.nms_time)."""
    if not len(rows):
        return True
    dist = np.maximum(rows[:, 0] - b, a - rows[:, 1])
    return bool((dist >= gap).all())


def _overlaps(a, b, iv):
    if not len(iv):
        return False
    return bool(((iv[:, 0] < b) & (iv[:, 1] > a)).any())


def seg_mask(S: dict, tau: float) -> np.ndarray:
    return S["gate"] >= tau


def propose(d: dict, rule: str, prm: dict, postproc) -> pd.DataFrame:
    """Extra detector intervals (stream_id, t_start, t_end, score, t_alarm) for every stream of d."""
    thr, smooth, min_d, gap = postproc
    out = []
    for sid, S in d["streams"].items():
        p = S["p"]
        rows = d["rows"][d["rows"].stream_id == sid][["t_start", "t_end"]].values
        cb = d["cands_base"][d["cands_base"].stream_id == sid][["t_start", "t_end"]].values
        m = seg_mask(S, prm.get("tau", 0.0))
        new = []
        if rule == "sec":
            ev = to_events(sid, np.where(m, p, 0.0), prm["thr2"], smooth, prm["min_d2"], gap)
            for _sid, a, b, sc, _al in ev:
                if sc >= thr and not prm.get("allow_above", False):
                    pass   # a region above the base threshold that the base decode dropped (min-dur): allowed
                if _overlaps(a, b, cb) or not _far(a, b, rows, prm["nms"]):
                    continue
                new.append((a, b, sc))
        elif rule == "amax":
            for a0, b0 in S["segs"]:
                if not m[a0]:
                    continue
                ta, tb = a0 * STEP, b0 * STEP
                if _overlaps(ta, tb, rows):
                    continue
                seg = p[a0:b0]
                pk = float(seg.max())
                if pk < prm["pmin"]:
                    continue
                i = int(np.argmax(seg))
                lv = prm["frac"] * pk
                lo = i
                while lo > 0 and seg[lo - 1] >= lv:
                    lo -= 1
                hi = i
                while hi < len(seg) - 1 and seg[hi + 1] >= lv:
                    hi += 1
                a, b = (a0 + lo) * STEP, (a0 + hi + 1) * STEP
                if b - a < prm.get("dmin", 1.0):   # widen symmetrically to the duration floor
                    c = 0.5 * (a + b)
                    a, b = max(ta, c - prm.get("dmin", 1.0) / 2), min(tb, c + prm.get("dmin", 1.0) / 2)
                if not _far(a, b, rows, prm["nms"]):
                    continue
                new.append((a, b, pk))
        elif rule == "raw":   # low-gate segments: decode the UNGATED blend at a (higher) threshold thr_r
            lowm = S["gate"] < prm["tau_lo"]
            ev = to_events(sid, np.where(lowm, S["raw"], 0.0), prm["thr_r"], smooth, prm.get("min_d2", min_d), gap)
            for _sid, a, b, sc, _al in ev:
                if _overlaps(a, b, cb) or not _far(a, b, rows, prm["nms"]):
                    continue
                new.append((a, b, sc))
        else:
            raise ValueError(rule)
        for a, b, sc in new:
            out.append((sid, a, b, float(sc), a))
    return events_df(out) if out else pd.DataFrame(columns=COLS)


def add_rows(d: dict, extra: pd.DataFrame, scale: float = 0.1, dmin: float = 1.0, mode: str = "det",
             bounds: dict | None = None, nms: float = 4.0) -> pd.DataFrame:
    """Base rows + extra rows. Boundaries of the extra rows: mode 'det' (detector interval), 'rep' (base replay
    stage on the interval), 'seg' (annotator on the whole junction segment), each with the 1.0 s duration floor
    and clipped to the stream. Extra rows get score = scale * peak (< every base row's score, so they never take a
    match from a base row in the evaluator's score-ordered matching), then the base 4 s time NMS runs on the union
    (base rows are processed first and are never removed)."""
    if not len(extra):
        return d["rows"].copy()
    e = extra.copy()
    if mode != "det":
        k = [(r.stream_id, r.t_start, r.t_end) for r in e.itertuples()]
        b = [bounds[x] for x in k]
        e["t_start"] = [x[f"{mode}_s"] for x in b]
        e["t_end"] = [x[f"{mode}_e"] for x in b]
    e["t_start"] = np.maximum(e.t_start.values, 0.0)
    e["t_end"] = np.maximum(e.t_end.values, e.t_start.values + dmin)
    e["t_end"] = [min(t, d["durations"][s]) for s, t in zip(e.stream_id, e.t_end)]
    e = e[e.t_end > e.t_start]
    e["score"] = np.round(e.score.values * scale, 6)
    e["t_alarm"] = e.t_start
    e = e.round(3)
    out = pd.concat([d["rows"], e[COLS]], ignore_index=True)
    out = CL.nms_time(out, nms)
    return out.sort_values(["stream_id", "t_start"], kind="stable").reset_index(drop=True)


def rank(rep: dict) -> float:
    return rep["ranking_score"]["f1_mean"]


def counts(rep: dict) -> tuple:
    b = rep["by_theta"]["0.3"]
    return b["tp"], b["fp"], b["fn"]


GRIDS = {
    "raw": [dict(tau_lo=t, thr_r=r, min_d2=m, nms=4.0)
            for t, r, m in itertools.product([0.1, 0.2, 0.3, 0.5], [0.3, 0.4, 0.5, 0.6, 0.7], [0.4, 0.8])],
    "sec": [dict(tau=t, thr2=r, min_d2=m, nms=4.0)
            for t, r, m in itertools.product([0.3, 0.5, 0.7, 0.8, 0.9, 0.95], [0.1, 0.15, 0.2, 0.25, 0.3],
                                             [0.2, 0.4, 0.6, 0.8])],
    "amax": [dict(tau=t, pmin=q, frac=f, nms=4.0, dmin=1.0)
             for t, q, f in itertools.product([0.3, 0.5, 0.7, 0.8, 0.9, 0.95], [0.1, 0.15, 0.2, 0.25, 0.3],
                                              [0.5, 0.7])],
    # idea (a) literally: any segment with high P(fall clip) and no row gets the annotator's output on the segment
    # crop (rule amax with a near-zero / zero peak requirement; used with boundary mode 'seg')
    "segcrop": [dict(tau=t, pmin=q, frac=0.5, nms=4.0, dmin=1.0)
                for t, q in itertools.product([0.5, 0.7, 0.8, 0.9, 0.95, 0.98], [0.0, 0.05, 0.1])],
}


def tune(d: dict, rule: str, modes=("det",), bounds=None, grid=None):
    if isinstance(grid, str):
        grid = GRIDS[grid]
    """Grid on the bundle's own training OOF (official evaluator, in process). Returns (table, base score)."""
    gt = gt_train(d)
    base = score_inproc(d["rows"], gt, d["durations"])
    tab = []
    for prm in grid or GRIDS[rule]:
        ex = propose(d, rule, prm, d["postproc"])
        for mode in modes:
            rows = add_rows(d, ex, mode=mode, bounds=bounds)
            rep = score_inproc(rows, gt, d["durations"])
            tab.append({**prm, "mode": mode, "n_add": len(rows) - len(d["rows"]), "rank": rank(rep),
                        "delta": rank(rep) - rank(base), "tp_fp_fn": counts(rep)})
    return pd.DataFrame(tab), base


def select(tab: pd.DataFrame) -> dict | None:
    """Pre-registered train-only choice: the config with the largest training-OOF delta; ties -> fewest added
    rows, then grid order. None (= base system unchanged) when no config improves the training OOF."""
    t = tab.reset_index(drop=True).assign(_i=lambda x: np.arange(len(x)))
    t = t.sort_values(["delta", "n_add", "_i"], ascending=[False, True, True])
    best = t.iloc[0]
    if best.delta <= 1e-12:
        return None
    return {k: v for k, v in best.items() if k not in ("n_add", "rank", "delta", "tp_fp_fn", "_i")}

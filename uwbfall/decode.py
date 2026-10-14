"""Probability sequence -> candidate events (prototype post-processing) and its OOF operating-point search."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .paths import STEP_S
from .scoring import COLS, score_inproc

STEP = STEP_S


def to_events(sid: str, p: np.ndarray, thr: float, smooth: int, min_d: float, gap: float) -> list[tuple]:
    """Port of prototype_detector.to_events (block index a -> time a*0.2 s)."""
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


def events_df(evs: list[tuple]) -> pd.DataFrame:
    """Same rounding as prototype write_sub (3 decimals) -- candidates are exchanged at this precision."""
    return pd.DataFrame(evs, columns=COLS).round(3)


def decode_streams(probs: dict[str, np.ndarray], g) -> pd.DataFrame:
    evs = []
    for sid, p in probs.items():
        evs += to_events(sid, p, *g)
    return events_df(evs)


def grid(cfg: dict) -> list[tuple]:
    G = cfg["decode"]["grid"]
    return list(itertools.product(G["thr"], G["smooth"], G["min_dur"], G["gap"]))


def tune(probs: dict[str, np.ndarray], gt: list[dict], durations: dict[str, float], cfg: dict, log=None):
    """Operating point maximising the official ranking F1 on (OOF) probabilities. First maximum wins."""
    if cfg["decode"]["fixed"] is not None:
        return None, tuple(cfg["decode"]["fixed"]), []
    best, table = None, []
    for g in grid(cfg):
        r = score_inproc(decode_streams(probs, g), gt, durations)
        sc = r["ranking_score"]["f1_mean"]
        table.append([*g, sc])
        if best is None or sc > best[0]:
            best = (sc, g)
    if log:
        log(f"  operating point {best[1]} OOF rank F1 {best[0]:.4f}")
    return best[0], best[1], table

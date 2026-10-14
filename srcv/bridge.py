"""Bridge to the uwbfall pipeline package (/home/user/challenge1/uwbfall).

    from srcv.bridge import pipeline_sources, seed_uwbfall_cache
    tr = pipeline_sources(k, "train")      # list[uwbfall.VirtualSource], ids cv{k}_train_XX
    te = pipeline_sources(k, "test")
    seed_uwbfall_cache(k, "test")          # optional: copy srcv's cached features into uwbfall's cache

``pipeline_composition`` returns composition rows with FULL-PRECISION source_t0/source_t1 and the event columns
(event_t0, event_t1, fall_type, quality, needs_review, ignore) that uwbfall.sources_from_composition reads,
so no half-frame nudging is needed (resolve_half_frames=False).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import core as C

if "/home/user/challenge1" not in sys.path:
    sys.path.insert(0, "/home/user/challenge1")

COMP_COLS = ["stream_id", "segment_id", "kind", "path", "label", "dataset", "t_start", "t_end", "source_t0",
             "source_t1", "event_t0", "event_t1", "fall_type", "quality", "needs_review", "ignore", "subject"]


def pipeline_composition(fold: int, part: str) -> pd.DataFrame:
    c = C._comp(fold, part).copy()
    c["source_t0"] = c.source_t0_exact.astype(float)
    c["source_t1"] = c.source_t1_exact.astype(float)
    c["event_t0"] = pd.to_numeric(c.event_t0.replace("", np.nan))
    c["event_t1"] = pd.to_numeric(c.event_t1.replace("", np.nan))
    c["t_start"] = c.t_start.astype(float)
    c["t_end"] = c.t_end.astype(float)
    return c[COMP_COLS]


def pipeline_sources(fold: int, part: str) -> list:
    import uwbfall

    out = uwbfall.sources_from_composition(pipeline_composition(fold, part), resolve_half_frames=False)
    dur = C.durations(fold, part)
    for s in out:
        if abs(s.duration - dur[s.sid]) > 1e-9:
            raise RuntimeError(f"{s.sid}: uwbfall n_frames {s.n_frames} != srcv {dur[s.sid] * 50}")
    return out


def seed_uwbfall_cache(fold: int, part: str, cfg: dict | None = None) -> int:
    """Write uwbfall's feature-cache files from srcv's cache (same extractor maths; subset of keys). Returns n."""
    from uwbfall import extract as UX

    n = 0
    srcs = {s.stream_id: s for s in C.sources(fold, part)}
    for ps in pipeline_sources(fold, part):
        p = UX.cache_path(ps, cfg)
        if p.exists():
            continue
        F = C.load_feats(srcs[ps.sid], compute=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, sid=np.array(ps.sid), **{k: F[k] for k in UX.KEYS})
        tmp.replace(p)
        n += 1
    return n

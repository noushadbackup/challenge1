"""Bridge to the uwbfall pipeline package (/home/user/challenge1/uwbfall).

    from srcv.bridge import pipeline_sources, seed_uwbfall_cache
    tr = pipeline_sources(k, "train")      # list[uwbfall.VirtualSource], ids cv{k}_train_XX
    te = pipeline_sources(k, "test")
    seed_uwbfall_cache(k, "test")          # optional: copy srcv's cached features into uwbfall's cache
    spec(k, "test")                        # 'comp:<csv>:<events.csv>@cv{k}_test_' for uwbfall.parse_sources / CLI

``pipeline_composition`` returns composition rows with FULL-PRECISION source_t0/source_t1 and the event columns
(event_t0, event_t1, fall_type, quality, needs_review, ignore) that uwbfall.sources_from_composition reads,
so no half-frame nudging is needed (resolve_half_frames=False).
"""
from __future__ import annotations

import sys
from pathlib import Path

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


def pipeline_csv(fold: int, part: str) -> "Path":
    """fold_k/{part}_pipeline_composition.csv: pipeline_composition() written with repr() floats (written once)."""
    p = C.fold_dir(fold) / f"{part}_pipeline_composition.csv"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        pipeline_composition(fold, part).to_csv(tmp, index=False, float_format=None)
        tmp.replace(p)
    return p


def spec(fold: int, part: str) -> str:
    """uwbfall source spec of a fold part, usable by uwbfall.parse_sources and `python3 -m uwbfall ... --sources`.

    Note: uwbfall.parse_sources reads floats with pandas' default (fast) parser, which is off by 1 ulp on ~7 % of
    the repr() times. verify.v8_spec shows this changes no frame index (same T, slices, events, composition), so
    the streams are bit-identical to pipeline_sources(); only uwbfall's content-key (feature-cache key) differs,
    i.e. features are cached once more under the spec's key. pipeline_sources() is the exact path.
    """
    return f"comp:{pipeline_csv(fold, part)}:{C.events_csv(fold, part)}@cv{fold}_{part}_"


def pipeline_sources(fold: int, part: str) -> list:
    import uwbfall

    out = uwbfall.sources_from_composition(pipeline_composition(fold, part), resolve_half_frames=False)
    dur = C.durations(fold, part)
    for s in out:
        if abs(s.duration - dur[s.sid]) > 1e-9:
            raise RuntimeError(f"{s.sid}: uwbfall n_frames {s.n_frames} != srcv {dur[s.sid] * 50}")
    return out


def seed_uwbfall_cache(fold: int, part: str, cfg: dict | None = None, with_spec: bool = False) -> int:
    """Write uwbfall's feature-cache files from srcv's cache (same extractor maths; subset of keys). Returns n.
    with_spec: also seed the cache keys of uwbfall.parse_sources(spec(fold, part)) (same streams, verify V8;
    ~28 MB per stream more disk, so off by default)."""
    import uwbfall
    from uwbfall import extract as UX

    n = 0
    srcs = {s.stream_id: s for s in C.sources(fold, part)}
    todo = pipeline_sources(fold, part)
    if with_spec:
        todo += uwbfall.parse_sources(spec(fold, part))
    for ps in todo:
        p = UX.cache_path(ps, cfg)
        if p.exists():
            continue
        F = C.load_feats(srcs[ps.sid], compute=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, sid=np.array(ps.sid), **{k: F[k] for k in UX.KEYS})
        tmp.replace(p)
        n += 1
    return n

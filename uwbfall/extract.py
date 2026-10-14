"""Per-frame / per-block raw features, one radar at a time, with an on-disk cache.

Port of work/E1_pseudo_test/e1lib._one_radar (bit-exact replica of analysis/extract_features.process),
restricted to the keys the pipeline uses: e_raw, e_mot, e_diff (3, T) float32; mot_map (3, T//10, 184) float16;
prof_02 (3, T//10, 368) float16 (= 1000 x mean raw CIR per 0.2 s block).

Cache: <cache_dir>/feat_<sha256(content_key + EXTRACTOR_VERSION)[:24]>.npz (stream content hash + version).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .paths import RATE_HZ, cache_dir
from .sources import StreamSource

EXTRACTOR_VERSION = "uwbfall-extract-v1(e1lib._one_radar subset: e_raw,e_mot,e_diff,mot_map,prof_02)"
TAU = 3.0
KEYS = ("e_raw", "e_mot", "e_diff", "mot_map", "prof_02")


def _block_mean(x: np.ndarray, size: int) -> np.ndarray:
    n = x.shape[1] // size
    return x[:, : n * size].reshape(x.shape[0], n, size, *x.shape[2:]).mean(axis=2)


def _one_radar(cir32: np.ndarray) -> dict:
    """cir32: (1, T, 368) float32. Same maths and dtypes as extract_features.process."""
    cir = cir32.astype(np.float64)
    alpha = np.exp(-1.0 / (RATE_HZ * TAU))
    b, a = [1.0 - alpha], [1.0, -alpha]
    zi = lfilter_zi(b, a)[0] * cir[:, :1, :]
    avg, _ = lfilter(b, a, cir, axis=1, zi=zi)
    res = (cir - avg).astype(np.float32)
    del avg
    out = {
        "e_raw": (cir**2).sum(2).astype(np.float32),
        "e_mot": (res.astype(np.float64) ** 2).sum(2).astype(np.float32),
    }
    diff = np.diff(cir, axis=1, prepend=cir[:, :1])
    out["e_diff"] = (diff**2).sum(2).astype(np.float32)
    del diff
    out["mot_map"] = _block_mean(np.hypot(res[..., :184], res[..., 184:]), 10).astype(np.float16)
    del res
    out["prof_02"] = (1e3 * _block_mean(cir, 10)).astype(np.float16)
    return out


def extract(source: StreamSource) -> dict:
    parts = [_one_radar(np.asarray(source.radar(r))) for r in range(3)]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def cache_path(source: StreamSource, cfg: dict | None = None) -> Path:
    key = hashlib.sha256((source.content_key() + "|" + EXTRACTOR_VERSION).encode()).hexdigest()[:24]
    return cache_dir(cfg) / f"feat_{key}.npz"


def get_features(source: StreamSource, cfg: dict | None = None, log=None) -> dict:
    use_cache = (cfg or {}).get("runtime", {}).get("use_cache", True)
    p = cache_path(source, cfg) if use_cache else None
    if p is not None and p.exists():
        with np.load(p) as z:
            return {k: z[k] for k in KEYS}
    F = extract(source)
    if p is not None:
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, sid=np.array(source.sid), **F)
        tmp.replace(p)
        if log:
            log(f"  cached features {source.sid} -> {p.name}")
    return F

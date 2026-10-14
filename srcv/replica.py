"""Bit-exact replicas of the package stream writer and of the team feature extractor.

Vendored from work/E1_pseudo_test/e1lib.py (E1 verified them on the 5 real val streams); re-verified
by ``python3 -m srcv.verify`` in this package (train + val streams, see README).

- ``write_stream``  : line-by-line replica of scripts/build_streams._write_stream without the h5 write
                      (0.2 s crossfade, event protection, same rounding). Returns cir + events + composition.
- ``plan_stream``   : the same index arithmetic run on segment *lengths only* (no radar data is read),
                      used to produce composition/events CSVs instantly. Verified equal to ``write_stream``.
- ``extract``       : replica of analysis/extract_features.process, one radar at a time (same 7 keys/dtypes).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.signal import lfilter, lfilter_zi  # noqa: E402

PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
if str(PKG / "src") not in sys.path:
    sys.path.insert(0, str(PKG / "src"))

from abc2027.streams import FADE_S  # noqa: E402

RATE_HZ = 50.0
STEP_S = 1.0 / RATE_HZ
FADE_N = int(round(FADE_S * RATE_HZ))
TAU = 3.0


def seg_file(relative: str) -> Path:
    return PKG / "segments" / (relative.replace("/", "__") + ".npy")


@lru_cache(maxsize=None)
def seg_len(relative: str) -> int:
    """Number of 50 Hz frames of an aligned package segment (mmap header only)."""
    return int(np.load(seg_file(relative), mmap_mode="r").shape[1])


def _get(segment, name, default=None):
    return segment[name] if isinstance(segment, dict) else getattr(segment, name, default)


def _indices(relative: str, source_t0: float, source_t1: float) -> tuple[int, int]:
    n = seg_len(relative)
    i0 = int(np.clip(np.round(source_t0 / STEP_S), 0, n - 1))
    i1 = int(np.clip(np.round(source_t1 / STEP_S), i0 + 1, n))
    return i0, i1


def _rows(stream_id, order, segment, origin, cursor, i0):
    comp = {
        "stream_id": stream_id, "segment_id": f"{stream_id}_{order:04d}", "kind": _get(segment, "kind"),
        "path": _get(segment, "path"), "label": _get(segment, "label"), "dataset": _get(segment, "dataset"),
        "t_start": f"{origin * STEP_S:.3f}", "t_end": f"{cursor * STEP_S:.3f}",
        "source_t0": f"{_get(segment, 'source_t0'):.3f}", "source_t1": f"{_get(segment, 'source_t1'):.3f}",
    }
    event = None
    if _get(segment, "kind") == "fall":
        e0, e1 = _get(segment, "event_t0"), _get(segment, "event_t1")
        event_a = origin + int(np.round(e0 / STEP_S)) - i0
        event_b = origin + int(np.round(e1 / STEP_S)) - i0
        event = {
            "stream_id": stream_id, "event_id": f"{stream_id}_{order:04d}",
            "t_start": f"{event_a * STEP_S:.3f}", "t_end": f"{event_b * STEP_S:.3f}",
            "fall_type": _get(segment, "fall_type"), "ignore": str(bool(_get(segment, "ignore"))).lower(),
            "quality": _get(segment, "quality"), "needs_review": str(bool(_get(segment, "needs_review"))).lower(),
            "source_path": _get(segment, "path"), "source_t_start": f"{e0:.3f}",
            "source_t_end": f"{e1:.3f}", "dataset": _get(segment, "dataset"),
            "_event_b": event_b,
        }
    return comp, event


def plan_stream(stream_id: str, segments) -> tuple[int, list[dict], list[dict]]:
    """Index arithmetic of _write_stream on lengths only. Returns (n_frames, events, composition)."""
    chunk_lens: list[int] = []
    events, composition = [], []
    cursor = 0
    last_event_end = -1
    for order, segment in enumerate(segments):
        i0, i1 = _indices(_get(segment, "path"), _get(segment, "source_t0"), _get(segment, "source_t1"))
        L = i1 - i0
        blend = 0
        if chunk_lens:
            protect = L
            if _get(segment, "event_t0") is not None:
                protect = max(0, int(np.round(_get(segment, "event_t0") / STEP_S)) - i0)
            tail_room = cursor - last_event_end
            blend = min(FADE_N, L, chunk_lens[-1], protect, max(tail_room, 0))
        origin = cursor - blend
        if L > blend:
            chunk_lens.append(L - blend)
        cursor = origin + L
        comp, event = _rows(stream_id, order, segment, origin, cursor, i0)
        composition.append(comp)
        if event is not None:
            last_event_end = max(last_event_end, event.pop("_event_b"))
            events.append(event)
    return cursor, events, composition


def _slice(relative: str, source_t0: float, source_t1: float):
    cir = np.load(seg_file(relative), mmap_mode="r")
    i0, i1 = _indices(relative, source_t0, source_t1)
    return np.array(cir[:, i0:i1], dtype=np.float32), i0, i1


def write_stream(stream_id: str, segments) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Replica of build_streams._write_stream (in memory). Returns cir (3, T, 368) float32, events, composition."""
    chunks: list[np.ndarray] = []
    events, composition = [], []
    cursor = 0
    last_event_end = -1
    for order, segment in enumerate(segments):
        block, i0, _i1 = _slice(_get(segment, "path"), _get(segment, "source_t0"), _get(segment, "source_t1"))
        blend = 0
        if chunks:
            protect = block.shape[1]
            if _get(segment, "event_t0") is not None:
                event_index = int(np.round(_get(segment, "event_t0") / STEP_S)) - i0
                protect = max(0, event_index)
            tail_room = cursor - last_event_end
            blend = min(FADE_N, block.shape[1], chunks[-1].shape[1], protect, max(tail_room, 0))
            if blend > 0:
                weight = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
                head = chunks[-1][:, -blend:]
                chunks[-1][:, -blend:] = head * (1.0 - weight) + block[:, :blend] * weight
        origin = cursor - blend
        if block.shape[1] > blend:
            chunks.append(np.array(block[:, blend:], dtype=np.float32))
        cursor = origin + block.shape[1]
        comp, event = _rows(stream_id, order, segment, origin, cursor, i0)
        composition.append(comp)
        if event is not None:
            last_event_end = max(last_event_end, event.pop("_event_b"))
            events.append(event)
    cir = np.concatenate(chunks, axis=1) if chunks else np.zeros((3, 1, 368), dtype=np.float32)
    return cir, events, composition


# ---------------------------------------------------------------- feature extractor replica
def _block_mean(x: np.ndarray, size: int) -> np.ndarray:
    n = x.shape[1] // size
    return x[:, : n * size].reshape(x.shape[0], n, size, *x.shape[2:]).mean(axis=2)


def _one_radar(cir32: np.ndarray) -> dict:
    """cir32: (1, T, 368) float32. Same maths as analysis/extract_features.process on one radar."""
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
    out["amp_1s"] = np.abs(_block_mean(cir[..., :184] + 1j * cir[..., 184:], 50)).astype(np.float32)
    out["prof_1s"] = _block_mean(cir, 50).astype(np.float32)
    out["prof_02"] = (1e3 * _block_mean(cir, 10)).astype(np.float16)
    return out


def extract(get_radar) -> dict:
    """get_radar(r) -> (1, T, 368) float32 array. Returns the 7 cached keys, shape (3, ...)."""
    parts = [_one_radar(get_radar(r)) for r in range(3)]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}

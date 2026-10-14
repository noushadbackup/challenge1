"""Shared code for the E1 pseudo-test bank.

- val_groups():      read-only copy of scripts/build_streams._groups() restricted to val
- build_stream():    in-memory replica of scripts/build_streams._write_stream (0.2 s crossfade,
                     event protection, same index arithmetic); returns cir + events + composition
- extract():         low-RAM (per-radar) replica of analysis/extract_features.process; same keys,
                     same dtypes, verified bit-identical on real val streams (check_parity.py)

Never writes into the package. Run with PYTHONDONTWRITEBYTECODE=1 (also set below).
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.signal import lfilter, lfilter_zi  # noqa: E402

PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
OUT = Path("/home/user/data_raw/pseudo")
sys.path.insert(0, str(PKG / "src"))

from abc2027.streams import FADE_S  # noqa: E402

RATE_HZ = 50.0
STEP_S = 1.0 / RATE_HZ
FADE_N = int(round(FADE_S * RATE_HZ))
TAU = 3.0


def seg_file(relative: str) -> Path:
    return PKG / "segments" / (relative.replace("/", "__") + ".npy")


# ---------------------------------------------------------------- planning input
def val_groups() -> dict:
    """Same rows/fields as build_streams._groups()['val'] (copied logic, no package writes)."""
    events_list = list(csv.DictReader((PKG / "streams_events.csv").open(encoding="utf-8")))
    comp_list = list(csv.DictReader((PKG / "streams_composition.csv").open(encoding="utf-8")))
    events = {row["source_path"]: row for row in events_list}
    group = {"falls": [], "adl": [], "others": []}
    seen = set()
    for row in comp_list:
        path = row["path"]
        if path in seen:
            continue
        seen.add(path)
        if row["stream_id"].split("_")[0] != "val":
            continue
        dest = seg_file(path)
        if not dest.exists():
            continue
        n = np.load(dest, mmap_mode="r").shape[1]
        if row["kind"] == "adl":
            row["duration_s"] = n * 0.02
            group["adl"].append(row)
        elif row["kind"] == "fall":
            event = events.get(path)
            if event is None:
                continue
            event = dict(event)
            event["t_start"] = event["source_t_start"]
            event["t_end"] = event["source_t_end"]
            event["t_last"] = n * 0.02
            event["path"] = path
            group["falls"].append(event)
        elif row["kind"] == "other":
            row["duration_s"] = n * 0.02
            group["others"].append(row)
    return group


# ---------------------------------------------------------------- stream writer replica
def _slice(relative: str, source_t0: float, source_t1: float):
    cir = np.load(seg_file(relative), mmap_mode="r")
    i0 = int(np.clip(np.round(source_t0 / STEP_S), 0, cir.shape[1] - 1))
    i1 = int(np.clip(np.round(source_t1 / STEP_S), i0 + 1, cir.shape[1]))
    return np.array(cir[:, i0:i1], dtype=np.float32), i0, i1


def build_stream(stream_id: str, segments) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Line-by-line replica of build_streams._write_stream without the h5 write.

    ``segments`` are abc2027.streams.Segment objects (or anything with the same attributes).
    """
    chunks: list[np.ndarray] = []
    events, composition = [], []
    cursor = 0
    last_event_end = -1
    for order, segment in enumerate(segments):
        block, i0, _i1 = _slice(segment.path, segment.source_t0, segment.source_t1)
        blend = 0
        if chunks:
            protect = block.shape[1]
            if segment.event_t0 is not None:
                event_index = int(np.round(segment.event_t0 / STEP_S)) - i0
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
        composition.append({
            "stream_id": stream_id, "segment_id": f"{stream_id}_{order:04d}", "kind": segment.kind,
            "path": segment.path, "label": segment.label, "dataset": segment.dataset,
            "t_start": f"{origin * STEP_S:.3f}", "t_end": f"{cursor * STEP_S:.3f}",
            "source_t0": f"{segment.source_t0:.3f}", "source_t1": f"{segment.source_t1:.3f}",
        })
        if segment.kind == "fall":
            event_a = origin + int(np.round(segment.event_t0 / STEP_S)) - i0
            event_b = origin + int(np.round(segment.event_t1 / STEP_S)) - i0
            last_event_end = max(last_event_end, event_b)
            events.append({
                "stream_id": stream_id, "event_id": f"{stream_id}_{order:04d}",
                "t_start": f"{event_a * STEP_S:.3f}", "t_end": f"{event_b * STEP_S:.3f}",
                "fall_type": segment.fall_type, "ignore": str(segment.ignore).lower(),
                "quality": segment.quality, "needs_review": str(segment.needs_review).lower(),
                "source_path": segment.path, "source_t_start": f"{segment.event_t0:.3f}",
                "source_t_end": f"{segment.event_t1:.3f}", "dataset": segment.dataset,
            })
    cir = np.concatenate(chunks, axis=1) if chunks else np.zeros((3, 1, 368), dtype=np.float32)
    return cir, events, composition


# ---------------------------------------------------------------- feature extractor replica
def _block_mean(x: np.ndarray, size: int) -> np.ndarray:
    n = x.shape[1] // size
    return x[:, : n * size].reshape(x.shape[0], n, size, *x.shape[2:]).mean(axis=2)


def _one_radar(cir32: np.ndarray) -> dict:
    """cir32: (1, T, 368) float32. Same maths as extract_features.process on one radar."""
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def package_manifest() -> str:
    """sha256 over every package file except __pycache__ (to prove we never write into it)."""
    h = hashlib.sha256()
    for p in sorted(PKG.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".h5" and p.suffix != ".npy":
            h.update(str(p.relative_to(PKG)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    # big binary files: name + size + mtime only (hashing 10 GB is too slow)
    for p in sorted(list(PKG.rglob("*.h5")) + list(PKG.rglob("*.npy"))):
        st = p.stat()
        h.update(f"{p.relative_to(PKG)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()

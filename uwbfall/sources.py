"""Stream sources: real stream_XX.h5 files, or "virtual" streams rebuilt in memory from composition rows.

Both expose the same small interface used by the pipeline:
  sid, n_frames, duration, content_key(), radar(r) -> (1, T, 368) float32, crop(i0, i1) -> (3, n, 368) float32,
  release() (drops any in-memory CIR).

The virtual writer is a port of work/E1_pseudo_test/e1lib.build_stream, itself a line-by-line replica of
scripts/build_streams._write_stream (0.2 s crossfade, event protection, same rounding). E1 verified it
bit-exact against all 5 published val streams. Nothing is ever written as h5.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from .paths import PKG, RATE_HZ

WRITER_VERSION = "e1-write-stream-v1"
STEP_S = 1.0 / RATE_HZ
FADE_S = 0.2  # abc2027.streams.FADE_S
FADE_N = int(round(FADE_S * RATE_HZ))


# ----------------------------------------------------------------------------- segments
@dataclass
class Seg:
    """Same attributes as abc2027.streams.Segment (which can be passed instead)."""
    kind: str
    path: str
    label: str = ""
    dataset: str = ""
    source_t0: float = 0.0
    source_t1: float = 0.0
    event_t0: float | None = None
    event_t1: float | None = None
    fall_type: str = ""
    quality: str = ""
    needs_review: bool = False
    ignore: bool = False


def seg_file(relative: str) -> Path:
    return PKG / "segments" / (relative.replace("/", "__") + ".npy")


def _slice_idx(n: int, source_t0: float, source_t1: float) -> tuple[int, int]:
    i0 = int(np.clip(np.round(source_t0 / STEP_S), 0, n - 1))
    i1 = int(np.clip(np.round(source_t1 / STEP_S), i0 + 1, n))
    return i0, i1


def build_stream(stream_id: str, segments, data: bool = True):
    """Port of e1lib.build_stream. Returns (cir or None, events, composition).

    ``data=False`` runs the identical index arithmetic on segment lengths only (no CIR), which is enough
    for events/composition/length (used for labels and cache keys without materialising the stream).
    """
    chunks: list[np.ndarray] = []
    chunk_lens: list[int] = []
    events, composition = [], []
    cursor = 0
    last_event_end = -1
    for order, segment in enumerate(segments):
        arr = np.load(seg_file(segment.path), mmap_mode="r")
        i0, i1 = _slice_idx(arr.shape[1], segment.source_t0, segment.source_t1)
        blen = i1 - i0
        block = np.array(arr[:, i0:i1], dtype=np.float32) if data else None
        blend = 0
        if chunk_lens:
            protect = blen
            if segment.event_t0 is not None:
                event_index = int(np.round(segment.event_t0 / STEP_S)) - i0
                protect = max(0, event_index)
            tail_room = cursor - last_event_end
            blend = min(FADE_N, blen, chunk_lens[-1], protect, max(tail_room, 0))
            if blend > 0 and data:
                weight = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
                head = chunks[-1][:, -blend:]
                chunks[-1][:, -blend:] = head * (1.0 - weight) + block[:, :blend] * weight
        origin = cursor - blend
        if blen > blend:
            chunk_lens.append(blen - blend)
            if data:
                chunks.append(np.array(block[:, blend:], dtype=np.float32))
        cursor = origin + blen
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
    n_frames = int(sum(chunk_lens)) if chunk_lens else 1
    if data:
        cir = np.concatenate(chunks, axis=1) if chunks else np.zeros((3, 1, 368), dtype=np.float32)
        assert cir.shape[1] == n_frames
        return cir, events, composition
    return None, events, composition, n_frames


# ----------------------------------------------------------------------------- sources
class StreamSource:
    sid: str
    n_frames: int

    @property
    def duration(self) -> float:
        return float(self.n_frames) / RATE_HZ

    def content_key(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def radar(self, r: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def crop(self, i0: int, i1: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:
        pass

    def events(self) -> pd.DataFrame | None:
        return None


_HASH_INDEX: dict | None = None


def _hash_index_path() -> Path:
    from .paths import DEFAULT_CACHE
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE / "h5_hash_index.json"


class H5Source(StreamSource):
    """A published stream file. Content key = sha256 of the cir dataset bytes (memoised by path/size/mtime)."""

    def __init__(self, path: str | Path, sid: str):
        self.path = Path(path).resolve()
        self.sid = sid
        with h5py.File(self.path, "r") as h:
            self.n_frames = int(h["cir"].shape[1])
        self._h = None

    def _file(self):
        if self._h is None:
            self._h = h5py.File(self.path, "r")
        return self._h

    def content_key(self) -> str:
        global _HASH_INDEX
        st = self.path.stat()
        memo = f"{self.path}|{st.st_size}|{st.st_mtime_ns}"
        idx_path = _hash_index_path()
        if _HASH_INDEX is None:
            try:
                _HASH_INDEX = json.loads(idx_path.read_text())
            except (OSError, ValueError):
                _HASH_INDEX = {}
        if memo not in _HASH_INDEX:
            h = hashlib.sha256()
            d = self._file()["cir"]
            h.update(f"{d.shape}|{d.dtype}".encode())
            step = 10000
            for r in range(d.shape[0]):
                for a in range(0, d.shape[1], step):
                    h.update(np.ascontiguousarray(d[r, a : a + step]).tobytes())
            _HASH_INDEX[memo] = "h5-" + h.hexdigest()
            tmp = idx_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_HASH_INDEX, indent=0))
            tmp.replace(idx_path)
        return _HASH_INDEX[memo]

    def radar(self, r: int) -> np.ndarray:
        return self._file()["cir"][r : r + 1]

    def crop(self, i0: int, i1: int) -> np.ndarray:
        return self._file()["cir"][:, i0:i1, :]

    def release(self) -> None:
        if self._h is not None:
            self._h.close()
            self._h = None


class VirtualSource(StreamSource):
    """A stream rebuilt in memory from segments (composition rows / abc2027.streams.Segment objects)."""

    def __init__(self, sid: str, segments: list):
        self.sid = sid
        self.segments = list(segments)
        _, ev, comp, n = build_stream(sid, self.segments, data=False)
        self._events, self._comp, self.n_frames = ev, comp, n
        self._cir = None

    def content_key(self) -> str:
        spec = []
        for s in self.segments:
            spec.append([s.kind, s.path, repr(float(s.source_t0)), repr(float(s.source_t1)),
                         None if s.event_t0 is None else repr(float(s.event_t0)),
                         None if s.event_t1 is None else repr(float(s.event_t1)), seg_file(s.path).stat().st_size])
        h = hashlib.sha256(json.dumps([WRITER_VERSION, spec]).encode())
        return "virt-" + h.hexdigest()

    def cir(self) -> np.ndarray:
        if self._cir is None:
            self._cir, _, _ = build_stream(self.sid, self.segments, data=True)
        return self._cir

    def radar(self, r: int) -> np.ndarray:
        return self.cir()[r : r + 1]

    def crop(self, i0: int, i1: int) -> np.ndarray:
        return self.cir()[:, i0:i1, :]

    def release(self) -> None:
        self._cir = None

    def events(self) -> pd.DataFrame:
        cols = ["stream_id", "event_id", "t_start", "t_end", "fall_type", "ignore", "quality", "needs_review",
                "source_path", "source_t_start", "source_t_end", "dataset"]
        return pd.DataFrame(self._events, columns=cols)

    def composition(self) -> pd.DataFrame:
        return pd.DataFrame(self._comp)


# ----------------------------------------------------------------------------- factories
def dir_sources(folder: str | Path, prefix: str | None = None) -> list[H5Source]:
    """stream_XX.h5 files in ``folder`` -> ids ``{prefix}_XX`` (prefix defaults to the folder name, as the
    official ``stream_durations``)."""
    folder = Path(folder)
    prefix = prefix or folder.name
    out = []
    for p in sorted(folder.glob("stream_*.h5")):
        out.append(H5Source(p, f"{prefix}_{int(p.stem.split('_')[1]):02d}"))
    if not out:
        raise FileNotFoundError(f"no stream_*.h5 under {folder}")
    return out


def split_sources(split: str) -> list[H5Source]:
    """A package split (train / val / test) -> H5 sources with ids train_XX / val_XX / test_XX."""
    return dir_sources(PKG / "streams" / split, split)


def sources_from_composition(comp: pd.DataFrame, events: pd.DataFrame | None = None,
                             stream_ids: Iterable[str] | None = None,
                             resolve_half_frames: bool = True) -> list[VirtualSource]:
    """Virtual sources from composition rows (columns as streams_composition.csv; full-precision source_t0/t1
    recommended). Fall rows need their event bounds: either ``event_t0/event_t1`` columns on ``comp`` or an
    events table with ``source_path, source_t_start, source_t_end, fall_type, quality, needs_review, ignore``.

    Caveat: the published CSVs print source times with 3 decimals, so a time on an exact half frame is
    ambiguous under np.round (E1 README). With ``resolve_half_frames`` (default) the ambiguity is resolved from
    the published stream times (see ``_resolve``); a residual ambiguity remains only when both ends of a
    non-fall row sit on half frames with the same length. Prefer full-precision rows for regenerated streams.
    """
    evmap = None
    if events is not None and "event_t0" not in comp.columns:
        evmap = events.drop_duplicates("source_path").set_index("source_path")
    out = []
    ids = list(stream_ids) if stream_ids is not None else list(pd.unique(comp.stream_id))
    for sid in ids:
        rows = comp[comp.stream_id == sid]
        if "segment_id" in rows.columns:
            rows = rows.sort_values("segment_id")
        segs = []
        for r in rows.itertuples(index=False):
            kw = {}
            if r.kind == "fall":
                if "event_t0" in comp.columns:
                    kw = dict(event_t0=float(r.event_t0), event_t1=float(r.event_t1),
                              fall_type=getattr(r, "fall_type", r.label), quality=getattr(r, "quality", ""),
                              needs_review=str(getattr(r, "needs_review", "false")).lower() == "true",
                              ignore=str(getattr(r, "ignore", "false")).lower() == "true")
                else:
                    if evmap is None:
                        raise ValueError("fall rows need event bounds (events table or event_t0/event_t1 columns)")
                    e = evmap.loc[r.path]
                    kw = dict(event_t0=float(e.source_t_start), event_t1=float(e.source_t_end), fall_type=e.fall_type,
                              quality=e.quality, needs_review=str(e.needs_review).lower() == "true",
                              ignore=str(e.ignore).lower() == "true")
            t0, t1 = float(r.source_t0), float(r.source_t1)
            if resolve_half_frames and hasattr(r, "t_start") and hasattr(r, "t_end"):
                t0, t1 = _resolve(r, t0, t1, kw, events, sid)
            segs.append(Seg(kind=r.kind, path=r.path, label=r.label if isinstance(r.label, str) else "",
                            dataset=r.dataset if isinstance(r.dataset, str) else "",
                            source_t0=t0, source_t1=t1, **kw))
        out.append(VirtualSource(sid, segs))
    return out


_NUDGES = [(0, 0), (-1e-6, -1e-6), (1e-6, 1e-6), (-1e-6, 0), (1e-6, 0), (0, -1e-6), (0, 1e-6), (-1e-6, 1e-6), (1e-6, -1e-6)]


def _resolve(r, t0: float, t1: float, kw: dict, events, sid: str) -> tuple[float, float]:
    """3-decimal CSVs: pick the +-1e-6 s nudge of (source_t0, source_t1) consistent with the published stream
    times -- block length = (t_end - t_start) / 0.02 for every row, and for fall rows the event offset inside
    the block = (event t_start - row t_start) / 0.02. (0, 0) wins ties. No raw stream data is used."""
    n_pub = int(round((float(r.t_end) - float(r.t_start)) / STEP_S))
    off_pub = None
    if r.kind == "fall" and events is not None and "t_start" in events.columns:
        e = events[(events.stream_id == sid) & (events.source_path == r.path)]
        if len(e) == 1:
            off_pub = int(round((float(e.t_start.iloc[0]) - float(r.t_start)) / STEP_S))
    n = np.load(seg_file(r.path), mmap_mode="r").shape[1]
    for d0, d1 in _NUDGES:
        i0, i1 = _slice_idx(n, t0 + d0, t1 + d1)
        if i1 - i0 != n_pub:
            continue
        if off_pub is not None and int(np.round(kw["event_t0"] / STEP_S)) - i0 != off_pub:
            continue
        return t0 + d0, t1 + d1
    return t0, t1


def parse_sources(spec: str) -> list[StreamSource]:
    """CLI source spec, comma separated. Each item is one of:
       train | val | test                      package split (streams/<split>/stream_XX.h5)
       dir:<folder>[:<prefix>]                 any folder of stream_XX.h5
       comp:<composition.csv>[:<events.csv>][@<id prefix>]   virtual streams rebuilt in memory
    """
    out: list[StreamSource] = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        if item.startswith("dir:"):
            parts = item[4:].split(":")
            out += dir_sources(parts[0], parts[1] if len(parts) > 1 else None)
        elif item.startswith("comp:"):
            body, _, pref = item[5:].partition("@")
            parts = body.split(":")
            comp = pd.read_csv(parts[0], dtype={"source_t0": float, "source_t1": float})
            ev = pd.read_csv(parts[1], dtype=str) if len(parts) > 1 else None
            ids = [s for s in pd.unique(comp.stream_id) if s.startswith(pref)] if pref else None
            out += sources_from_composition(comp, ev, ids)
        else:
            out += split_sources(item)
    return out


def events_for(sources: list[StreamSource], events_df: pd.DataFrame | None) -> pd.DataFrame:
    """Ground-truth events for the given sources: rows of ``events_df`` for their ids, else the virtual
    sources' own writer events."""
    sids = {s.sid for s in sources}
    parts = []
    if events_df is not None:
        parts.append(events_df[events_df.stream_id.isin(sids)])
    have = set(pd.concat(parts).stream_id) if parts else set()
    for s in sources:
        if s.sid not in have and isinstance(s, VirtualSource):
            parts.append(s.events())
    if not parts:
        raise ValueError("no ground-truth events for the training sources")
    return pd.concat(parts, ignore_index=True)

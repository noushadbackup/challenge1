"""SR-CV core: subject pool, folds, regenerated fold plans, stream sources (virtual or real h5).

Nothing here writes into the package. Regenerated streams are never stored as h5: a ``Source`` rebuilds its
cir on demand from composition rows (``replica.write_stream``) and caches only extracted features.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import replica as R  # noqa: E402
from abc2027.streams import Segment, finish_plan, prepare_split  # noqa: E402

PKG = R.PKG
HERE = Path(__file__).resolve().parent
DATA = Path("/home/user/data_raw/srcv")          # fold CSVs + feature cache (outside git)
FEATS = DATA / "feats"                           # regenerated-stream feature cache (E1 extractor keys)
REAL_FEATS = Path("/home/user/data_raw/feats")   # team cache for the real train/val streams
FOLDS_JSON = HERE / "folds.json"
DENSITY = 57.1864                                # configs/streams.yaml target_falls_per_hour
T_LAST_MODE = "n_minus_1"                        # see verify.py / README: matches 8/9 published clipped falls
PARTS = ("train", "test")
EXACT_COLS = ["source_t0_exact", "source_t1_exact", "event_t0", "event_t1", "fall_type", "quality",
              "needs_review", "ignore", "subject"]


# ---------------------------------------------------------------- subjects
def subject_of(path: str) -> str:
    """Fall subject (Test_P{d}_{n}) for falls/ Chute/Autre/Empty files; ADL_{n} for activities/<Act>_<n>.json."""
    if path.startswith("falls/"):
        return path.split("/")[2]
    m = re.search(r"_(\d+)\.json$", path)
    if not m:
        raise ValueError(f"cannot parse subject from {path}")
    return f"ADL_{int(m.group(1))}"


def subject_dataset(subject: str) -> str:
    return "ADL" if subject.startswith("ADL_") else f"D{subject.split('_')[1][1]}"


# ---------------------------------------------------------------- pool (train + val, package rows)
@lru_cache(maxsize=4)
def pool(t_last_mode: str = T_LAST_MODE) -> dict:
    """Planning rows for every train+val recording, in the format abc2027.streams.prepare_split expects.

    Same construction as scripts/build_streams._groups (fall rows from streams_events.csv, t_last from the
    segments/*.npy frame count), pooled over train and val, plus ``subject`` / ``origin`` keys.
    t_last_mode: 'n' -> n*0.02 (as _groups), 'n_minus_1' -> (n-1)*0.02 (as the published plan, see verify.py).
    """
    events = {r["source_path"]: r for r in csv.DictReader((PKG / "streams_events.csv").open(encoding="utf-8"))}
    out = {"falls": [], "adl": [], "others": []}
    seen = set()
    for row in csv.DictReader((PKG / "streams_composition.csv").open(encoding="utf-8")):
        path = row["path"]
        if path in seen:
            continue
        seen.add(path)
        n = R.seg_len(path)
        origin = row["stream_id"].split("_")[0]
        if row["kind"] == "fall":
            ev = dict(events[path])
            ev["t_start"] = ev["source_t_start"]
            ev["t_end"] = ev["source_t_end"]
            ev["t_last"] = (n - 1) * 0.02 if t_last_mode == "n_minus_1" else n * 0.02
            ev["path"] = path
            ev["subject"], ev["origin"] = subject_of(path), origin
            out["falls"].append(ev)
        elif row["kind"] == "adl":
            out["adl"].append({"path": path, "label": row["label"], "duration_s": n * 0.02,
                               "subject": subject_of(path), "origin": origin})
        elif row["kind"] == "other":
            out["others"].append({"path": path, "label": row["label"], "duration_s": n * 0.02,
                                  "position": row["dataset"].replace("Dataset ", ""),
                                  "subject": subject_of(path), "origin": origin})
    return out


def subjects_table() -> pd.DataFrame:
    """One row per subject: dataset, origin split, n falls, n evaluable under test rules, ADL seconds."""
    p = pool()
    rows = {}
    for f in p["falls"]:
        r = rows.setdefault(f["subject"], {"subject": f["subject"], "origin": f["origin"], "n_falls": 0, "n_eval": 0,
                                           "adl_s": 0.0})
        r["n_falls"] += 1
        r["n_eval"] += int(not (f["quality"] == "low" or str(f["needs_review"]).lower() == "true"))
    for a in p["adl"]:
        r = rows.setdefault(a["subject"], {"subject": a["subject"], "origin": a["origin"], "n_falls": 0, "n_eval": 0,
                                           "adl_s": 0.0})
        r["adl_s"] += a["duration_s"]
    t = pd.DataFrame(rows.values())
    t["dataset"] = t.subject.map(subject_dataset)
    return t.sort_values(["dataset", "subject"]).reset_index(drop=True)


# ---------------------------------------------------------------- folds
def folds() -> list[dict]:
    """The 4 SR-CV folds (srcv/folds.json): test subjects per dataset + ADL, seeds, stream ids."""
    return json.loads(FOLDS_JSON.read_text())["folds"]


def fold_subjects(fold: int, part: str) -> set[str]:
    f = folds()[fold]
    held = {s for group in f["test_subjects"].values() for s in group}
    if part == "test":
        return held
    if part == "train":
        return set(subjects_table().subject) - held
    raise ValueError(part)


def fold_dir(fold: int) -> Path:
    return DATA / f"fold_{fold}"


def _seed(fold: int, part: str) -> int:
    return int(folds()[fold]["seeds"][part])


def make_plan(fold: int, part: str) -> list[tuple[str, list[Segment]]]:
    """Official planning (prepare_split + finish_plan, density 57.1864) on the fold's subjects.

    part='train' -> split 'train' settings (adl_uses=3, no ignore flags);
    part='test'  -> split 'test' settings (adl_uses=2, low-quality / needs_review falls -> ignore=true).
    """
    subj = fold_subjects(fold, part)
    p = pool()
    falls = [r for r in p["falls"] if r["subject"] in subj]
    adl = [r for r in p["adl"] if r["subject"] in subj]
    others = [r for r in p["others"] if r["subject"] in subj]
    split = "test" if part == "test" else "train"
    bundle = prepare_split(split, falls, adl, others, rng=np.random.default_rng(_seed(fold, part)))
    plan = finish_plan(bundle, DENSITY)
    return [(f"cv{fold}_{part}_{i:02d}", st.segments) for i, st in enumerate(plan.streams)]


def generate(fold: int, part: str) -> dict:
    """Write fold_k/{part}_composition.csv and {part}_events.csv (no radar data is read)."""
    comp_rows, ev_rows, meta = [], [], []
    for sid, segs in make_plan(fold, part):
        n, events, comp = R.plan_stream(sid, segs)
        for c, s in zip(comp, segs):
            c.update({"source_t0_exact": repr(float(s.source_t0)), "source_t1_exact": repr(float(s.source_t1)),
                      "event_t0": "" if s.event_t0 is None else repr(float(s.event_t0)),
                      "event_t1": "" if s.event_t1 is None else repr(float(s.event_t1)),
                      "fall_type": s.fall_type, "quality": s.quality, "needs_review": str(s.needs_review).lower(),
                      "ignore": str(s.ignore).lower(), "subject": subject_of(s.path)})
        for e in events:
            e["subject"] = subject_of(e["source_path"])
        comp_rows += comp
        ev_rows += events
        meta.append({"stream_id": sid, "n_frames": n, "n_events": len(events),
                     "n_evaluable": sum(e["ignore"] == "false" for e in events)})
    d = fold_dir(fold)
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / f"{part}_composition.csv", comp_rows)
    _write_csv(d / f"{part}_events.csv", ev_rows)
    return {"streams": meta, "composition_sha256": _sha(d / f"{part}_composition.csv")}


def _write_csv(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- CSV accessors
@lru_cache(maxsize=32)
def _comp(fold: int, part: str) -> pd.DataFrame:
    return pd.read_csv(fold_dir(fold) / f"{part}_composition.csv", dtype=str, keep_default_na=False)


def composition(fold: int, part: str) -> pd.DataFrame:
    """Composition rows (official columns + exact source/event times, flags, subject). Times as float."""
    c = _comp(fold, part).copy()
    for k in ("t_start", "t_end", "source_t0_exact", "source_t1_exact"):
        c[k] = c[k].astype(float)
    return c


def events(fold: int, part: str) -> pd.DataFrame:
    """Events in the official streams_events.csv format (+ ``subject``). part='test' has ignore=true on flagged."""
    e = pd.read_csv(fold_dir(fold) / f"{part}_events.csv", dtype={"ignore": str, "needs_review": str})
    return e


def events_csv(fold: int, part: str) -> Path:
    return fold_dir(fold) / f"{part}_events.csv"


def durations(fold: int, part: str) -> dict[str, float]:
    """Stream durations exactly as abc2027.evaluate.stream_durations would read them from the h5 (T / 50)."""
    c = _comp(fold, part)
    last = c.groupby("stream_id", sort=True).t_end.apply(lambda s: s.astype(float).max())
    return {sid: float(int(round(t * R.RATE_HZ))) / R.RATE_HZ for sid, t in last.items()}


# ---------------------------------------------------------------- sources
@dataclass(frozen=True)
class Source:
    """One stream the pipeline can consume, virtual (rebuilt from composition rows) or real (h5).

    Fields (also available as a plain dict via ``as_dict()``):
      stream_id   str   e.g. 'cv0_test_03' (virtual) or 'val_02' (real)
      kind        str   'virtual' | 'h5'
      split       str   'train' | 'test' (virtual: generator settings; real: package split name)
      fold        int | None
      n_frames    int   T, cir has shape (3, T, 368); duration_s = T / 50
      duration_s  float
      segments    tuple[dict]  writer input rows (kind, path, label, dataset, source_t0, source_t1, event_t0,
                               event_t1, fall_type, quality, needs_review, ignore), full float precision.
                               Empty for real streams.
      h5_path     str   real streams only ('' for virtual)
      feats_path  str   E1/analysis feature npz (7 keys e_raw, e_mot, e_diff, mot_map, amp_1s, prof_1s, prof_02)
    """

    stream_id: str
    kind: str
    split: str
    fold: int | None
    n_frames: int
    duration_s: float
    segments: tuple = field(default=(), repr=False)
    h5_path: str = ""
    feats_path: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["segments"] = [dict(s) for s in self.segments]
        return d

    def load_cir(self) -> np.ndarray:
        return load_cir(self)

    def load_feats(self) -> dict:
        return load_feats(self)


def _segments_from_comp(rows: pd.DataFrame) -> tuple:
    segs = []
    for r in rows.itertuples(index=False):
        fall = r.kind == "fall"
        segs.append({
            "kind": r.kind, "path": r.path, "label": r.label, "dataset": r.dataset,
            "source_t0": float(r.source_t0_exact), "source_t1": float(r.source_t1_exact),
            "event_t0": float(r.event_t0) if fall else None, "event_t1": float(r.event_t1) if fall else None,
            "fall_type": r.fall_type, "quality": r.quality, "needs_review": r.needs_review == "true",
            "ignore": r.ignore == "true",
        })
    return tuple(segs)


def sources(fold: int, part: str) -> list[Source]:
    """Virtual stream sources of fold ``fold`` (0..3), part 'train' (regenerated training streams of the 18
    non-held-out fall subjects + 6 ADL subjects) or 'test' (held-out subjects, test settings)."""
    c = _comp(fold, part)
    dur = durations(fold, part)
    out = []
    for sid, rows in c.groupby("stream_id", sort=True):
        out.append(Source(stream_id=sid, kind="virtual", split="test" if part == "test" else "train", fold=fold,
                          n_frames=int(round(dur[sid] * R.RATE_HZ)), duration_s=dur[sid],
                          segments=_segments_from_comp(rows), feats_path=str(FEATS / f"{sid}.npz")))
    return out


def real_sources(split: str) -> list[Source]:
    """The published package streams (split 'train' | 'val' | 'test') as Sources (h5 kind)."""
    import h5py

    out = []
    for p in sorted((PKG / "streams" / split).glob("stream_*.h5")):
        sid = f"{split}_{int(p.stem.split('_')[1]):02d}"
        with h5py.File(p, "r") as h:
            T = int(h["cir"].shape[1])
        out.append(Source(stream_id=sid, kind="h5", split=split, fold=None, n_frames=T, duration_s=T / R.RATE_HZ,
                          h5_path=str(p), feats_path=str(REAL_FEATS / f"{sid}.npz")))
    return out


def real_events(split: str) -> pd.DataFrame:
    e = pd.read_csv(PKG / "streams_events.csv", dtype={"ignore": str, "needs_review": str})
    e = e[e.stream_id.str.startswith(split + "_")].copy()
    e["subject"] = e.source_path.map(subject_of)
    return e.reset_index(drop=True)


def _src(s) -> Source:
    return s if isinstance(s, Source) else Source(**{**s, "segments": tuple(s.get("segments", ()))})


def load_cir(source) -> np.ndarray:
    """(3, T, 368) float32 cir of a Source (or its dict). Virtual: bit-exact writer replica, ~0.4 GB / 30 min."""
    s = _src(source)
    if s.kind == "h5":
        import h5py

        with h5py.File(s.h5_path, "r") as h:
            return h["cir"][:]
    cir, _ev, _co = R.write_stream(s.stream_id, s.segments)
    if cir.shape[1] != s.n_frames:
        raise RuntimeError(f"{s.stream_id}: rebuilt T={cir.shape[1]} != planned {s.n_frames}")
    return cir


def plan_hash(source) -> str:
    """Hash of a virtual source's writer input (guards the feature cache against regenerated folds)."""
    s = _src(source)
    return hashlib.sha256(json.dumps([s.stream_id, s.n_frames, list(s.segments)], sort_keys=True).encode()).hexdigest()


def load_feats(source, compute: bool = True) -> dict:
    """Cached E1/analysis features (dict of 7 arrays, shape (3, ...)). Computes + caches if missing."""
    s = _src(source)
    p = Path(s.feats_path)
    if s.kind == "virtual" and p.exists():
        side = p.with_suffix(".json")
        if not side.exists() or json.loads(side.read_text()).get("plan_hash") != plan_hash(s):
            raise RuntimeError(f"stale feature cache for {s.stream_id}: delete {p} (plan changed)")
    if not p.exists():
        if not compute or s.kind == "h5":
            raise FileNotFoundError(p)
        cache_feats(s)
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def cache_feats(source) -> Path:
    s = _src(source)
    p = Path(s.feats_path)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    cir = load_cir(s)
    F = R.extract(lambda r: cir[r: r + 1])
    del cir
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **F)
    tmp.rename(p)
    p.with_suffix(".json").write_text(json.dumps({"plan_hash": plan_hash(s), "n_frames": s.n_frames}))
    return p

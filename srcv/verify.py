"""Verification of the SR-CV reconstruction and replicas against the published package.

    python3 -m srcv.verify [--bitexact train_00,val_04] [--fold-stream cv0_test_04]

V1  pool rows == official scripts/build_streams._groups() rows (train + val), field by field.
V2  public-seed replay: prepare_split(split, our rows) with the public seed reproduces the published fall clip
    windows (source_t0/source_t1) -> proves t_start/t_end/t_last and the rng consumption; compares t_last modes.
V3  plan_stream (lengths only) on the published composition reproduces every published composition time, event
    time and stream length (18 streams).
V4  write_stream bit-exact vs published h5 (selected streams) + vendored extractor bit-exact vs team cache.
V5  on a regenerated fold stream: write_stream composition/events/T == the stored fold CSVs (plan_stream).
V6  scorer: score_val(prototype val CSVs) == official CLI json, full report identical.
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import core as C  # noqa: E402
from . import replica as R  # noqa: E402
from . import scoring as S  # noqa: E402

sys.path.insert(0, str(C.PKG / "scripts"))
import build_streams as B  # noqa: E402  (read-only use of _groups)
from abc2027.streams import Segment, prepare_split  # noqa: E402

OUT = C.HERE / "results"
NUDGES = [(0, 0), (-1e-6, -1e-6), (1e-6, 1e-6), (-1e-6, 0), (1e-6, 0), (0, -1e-6), (0, 1e-6), (-1e-6, 1e-6),
          (1e-6, -1e-6)]


def v1_pool() -> dict:
    off = B._groups()
    res = {}
    for mode in ("n", "n_minus_1"):
        C.pool.cache_clear()
        mine = C.pool(mode)
        bad = {"falls": 0, "adl": 0, "others": 0, "t_last_diff_frames": []}
        for kind in ("falls", "adl", "others"):
            o = {r["path"]: r for s in ("train", "val") for r in off[s][kind]}
            m = {r["path"]: r for r in mine[kind]}
            assert set(o) == set(m), kind
            for p, r in o.items():
                keys = ["t_start", "t_end", "quality", "needs_review", "fall_type", "dataset", "path"] if kind == "falls" \
                    else ["path", "label", "duration_s"]
                if any(str(r[k]) != str(m[p][k]) for k in keys):
                    bad[kind] += 1
                if kind == "falls":
                    bad["t_last_diff_frames"].append(round((float(r["t_last"]) - float(m[p]["t_last"])) / 0.02))
        bad["t_last_diff_frames"] = {str(k): int(v) for k, v in zip(*np.unique(bad["t_last_diff_frames"],
                                                                               return_counts=True))}
        res[mode] = {**bad, "n": {k: len(v) for k, v in mine.items()}}
    C.pool.cache_clear()
    print("V1 pool vs official _groups():", json.dumps(res))
    return res


def v2_replay() -> dict:
    co = pd.read_csv(C.PKG / "streams_composition.csv")
    res = {}
    for mode in ("n", "n_minus_1"):
        C.pool.cache_clear()
        p = C.pool(mode)
        r = {}
        for split in ("train", "val"):
            sel = lambda rows: [x for x in rows if x["origin"] == split]  # noqa: E731
            b = prepare_split(split, sel(p["falls"]), sel(p["adl"]), sel(p["others"]))
            pub = co[(co.kind == "fall") & co.stream_id.str.startswith(split)].set_index("path")
            d0 = np.array([abs(pub.loc[s.path].source_t0 - s.source_t0) for s in b["falls"]])
            d1 = np.array([abs(pub.loc[s.path].source_t1 - s.source_t1) for s in b["falls"]])
            r[split] = {"n_falls": len(d0), "t0_mismatch": int((d0 > 6e-4).sum()), "t1_mismatch": int((d1 > 6e-4).sum()),
                        "max_abs_t1_diff": float(d1.max())}
        res[mode] = r
    C.pool.cache_clear()
    print("V2 public-seed replay of fall clip windows:", json.dumps(res))
    return res


def _published_segments(sid: str, cir_h5=None) -> list[Segment]:
    comp = pd.read_csv(C.PKG / "streams_composition.csv", dtype=str, keep_default_na=False)
    ev = pd.read_csv(C.PKG / "streams_events.csv", dtype=str).set_index("source_path")
    rows = comp[comp.stream_id == sid]
    segs = []
    for r in rows.itertuples():
        kw = {}
        if r.kind == "fall":
            e = ev.loc[r.path]
            kw = dict(event_t0=float(e.source_t_start), event_t1=float(e.source_t_end), fall_type=e.fall_type,
                      quality=e.quality, needs_review=e.needs_review == "true", ignore=e.ignore == "true")
        # 3-decimal CSV: a source time on an exact half frame is ambiguous under np.round; pick the +-1e-6 nudge
        # that reproduces the published block length (and, with data, the interior frames) as E1 did.
        n_pub = int(round((float(r.t_end) - float(r.t_start)) / R.STEP_S))
        o_pub = int(round(float(r.t_start) / R.STEP_S))
        t0, t1 = float(r.source_t0), float(r.source_t1)
        best = (t0, t1)
        for d0, d1 in NUDGES:
            i0, i1 = R._indices(r.path, t0 + d0, t1 + d1)
            if i1 - i0 != n_pub:
                continue
            if cir_h5 is not None:
                block, _, _ = R._slice(r.path, t0 + d0, t1 + d1)
                if not np.array_equal(cir_h5[0, o_pub + R.FADE_N: o_pub + n_pub - R.FADE_N],
                                      block[0, R.FADE_N: n_pub - R.FADE_N]):
                    continue
            best = (t0 + d0, t1 + d1)
            break
        segs.append(Segment(kind=r.kind, path=r.path, label=r.label, dataset=r.dataset, source_t0=best[0],
                            source_t1=best[1], **kw))
    return segs


def v3_plan_stream() -> dict:
    comp = pd.read_csv(C.PKG / "streams_composition.csv", dtype=str, keep_default_na=False)
    ev = pd.read_csv(C.PKG / "streams_events.csv", dtype=str, keep_default_na=False)
    out = {}
    for sid in sorted(comp.stream_id.unique()):
        split, i = sid.split("_")
        n, events, parts = R.plan_stream(sid, _published_segments(sid))
        with h5py.File(C.PKG / "streams" / split / f"stream_{i}.h5", "r") as h:
            T = h["cir"].shape[1]
        pc = comp[comp.stream_id == sid][["t_start", "t_end"]].reset_index(drop=True)
        mc = pd.DataFrame(parts)[["t_start", "t_end"]]
        cols = ["event_id", "t_start", "t_end", "fall_type", "ignore", "quality", "needs_review", "source_path",
                "source_t_start", "source_t_end", "dataset"]
        pe = ev[ev.stream_id == sid][cols].reset_index(drop=True)
        me = pd.DataFrame(events)[cols]
        out[sid] = {"T_equal": n == T, "composition_equal": bool(mc.equals(pc)), "events_equal": bool(me.equals(pe)),
                    "n_events": len(me)}
    ok = all(v["T_equal"] and v["composition_equal"] and v["events_equal"] for v in out.values())
    print(f"V3 plan_stream on {len(out)} published streams: all equal = {ok}; "
          f"events {sum(v['n_events'] for v in out.values())}")
    if not ok:
        print({k: v for k, v in out.items() if not (v["T_equal"] and v["composition_equal"] and v["events_equal"])})
    return {"all_equal": ok, "streams": out}


def v4_bitexact(sids: list[str]) -> dict:
    out = {}
    for sid in sids:
        split, i = sid.split("_")
        with h5py.File(C.PKG / "streams" / split / f"stream_{i}.h5", "r") as h:
            ref = h["cir"]
            segs = _published_segments(sid, ref)
            cir, _ev, _co = R.write_stream(sid, segs)
            same = ref.shape == cir.shape and all(np.array_equal(ref[r], cir[r]) for r in range(3))
        F = R.extract(lambda r: cir[r: r + 1])
        del cir
        with np.load(C.REAL_FEATS / f"{sid}.npz") as z:
            feq = {k: bool(z[k].dtype == F[k].dtype and np.array_equal(z[k], F[k])) for k in z.files}
        out[sid] = {"cir_bit_equal": bool(same), "features_bit_equal": feq}
        print(f"V4 {sid}: cir bit-equal {same}; extractor vs team cache {feq}", flush=True)
    return out


def v5_fold_stream(sid: str) -> dict:
    k, part = int(sid[2]), sid.split("_")[1]
    src = next(s for s in C.sources(k, part) if s.stream_id == sid)
    cir, events, parts = R.write_stream(sid, src.segments)
    co = C._comp(k, part)
    co = co[co.stream_id == sid][list(parts[0].keys())].reset_index(drop=True)
    ev = pd.read_csv(C.events_csv(k, part), dtype=str, keep_default_na=False)
    ev = ev[ev.stream_id == sid][list(events[0].keys())].reset_index(drop=True)
    res = {"T_equal": cir.shape[1] == src.n_frames, "composition_equal": bool(pd.DataFrame(parts).equals(co)),
           "events_equal": bool(pd.DataFrame(events).equals(ev)), "n_events": len(events), "T": int(cir.shape[1])}
    print(f"V5 regenerated {sid}: {res}")
    return res


def v6_scorer() -> dict:
    out = {}
    for p in [C.PKG / "sample_submission.csv"] + sorted(Path("/home/user/challenge1/reports/prototype").glob("val_*.csv")):
        mine = S.score_val(p)
        tmp = OUT / f"cli_{p.stem}.json"
        cmd = [sys.executable, "-B", "-m", "abc2027.evaluate", "--gt", "streams_events.csv", "--pred", str(p.resolve()),
               "--streams", "streams/val", "--ignore-flagged", "--out", str(tmp)]
        subprocess.run(cmd, check=True, cwd=C.PKG, capture_output=True,
                       env={"PYTHONPATH": str(C.PKG / "src"), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
        ref = json.loads(tmp.read_text())
        tmp.unlink()
        same = json.dumps(mine, sort_keys=True) == json.dumps(ref, sort_keys=True)
        out[p.name] = {"srcv": mine["ranking_score"]["f1_mean"], "cli": ref["ranking_score"]["f1_mean"], "identical": same}
    print("V6 score_val vs official CLI:", json.dumps(out))
    return out


def v7_bridge(sid: str = "cv0_test_04") -> dict:
    """uwbfall.VirtualSource built from srcv's composition == srcv source (n_frames, cir, features, events)."""
    from . import bridge
    from uwbfall import extract as UX

    nf_ok = True
    for k in range(4):
        for part in C.PARTS:
            try:
                bridge.pipeline_sources(k, part)
            except RuntimeError:
                nf_ok = False
    k, part = int(sid[2]), sid.split("_")[1]
    ps = {s.sid: s for s in bridge.pipeline_sources(k, part)}[sid]
    src = next(s for s in C.sources(k, part) if s.stream_id == sid)
    a, b = src.load_cir(), ps.cir()
    Fm, Fu = R.extract(lambda r: a[r: r + 1]), UX.extract(ps)
    me = C.events(k, part)
    me = me[me.stream_id == sid]
    ev = ps.events()
    res = {"n_frames_all_fold_parts_equal": nf_ok, "cir_equal": bool(np.array_equal(a, b)),
           "features_equal": {key: bool(np.array_equal(Fm[key], Fu[key])) for key in Fu},
           "events_equal": list(ev.t_start.astype(float)) == list(me.t_start) and list(ev.ignore) == list(me.ignore)}
    print(f"V7 uwbfall bridge on {sid}: {res}")
    return res


def main() -> None:
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:]
    if "--only-bridge" in args:
        (OUT / "verify_bridge.json").write_text(json.dumps(v7_bridge(), indent=1))
        return
    bitexact = ["val_04", "train_00"]
    fold_stream = "cv0_test_04"
    if "--bitexact" in args:
        bitexact = args[args.index("--bitexact") + 1].split(",")
    if "--fold-stream" in args:
        fold_stream = args[args.index("--fold-stream") + 1]
    rep = {"V1_pool": v1_pool(), "V2_public_seed_replay": v2_replay(), "V3_plan_stream": v3_plan_stream(),
           "V6_scorer": v6_scorer(), "V5_fold_stream": v5_fold_stream(fold_stream), "V4_bitexact": v4_bitexact(bitexact)}
    rep["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    (OUT / "verify.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"peak RSS {rep['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()

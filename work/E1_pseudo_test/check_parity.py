"""Verify the two replicas against the published data.

1. build_stream(): rebuild real val streams from streams_composition.csv (same segments, same order,
   same source windows) and compare with streams/val/stream_XX.h5 bit for bit, plus events CSV rows.
2. extract(): features of the real h5 vs the team cache /home/user/data_raw/feats/val_XX.npz.

Usage: python3 check_parity.py [idx ...]   (default: 4 0)
"""
from __future__ import annotations

import resource
import sys

import e1lib as L  # noqa: I001  (sets dont_write_bytecode first)
import h5py
import numpy as np
import pandas as pd
from abc2027.streams import Segment

FEATS = L.Path("/home/user/data_raw/feats")


def main() -> None:
    idxs = [int(a) for a in sys.argv[1:] if a.isdigit()] or [4, 0]
    comp = pd.read_csv(L.PKG / "streams_composition.csv", dtype=str)
    ev = pd.read_csv(L.PKG / "streams_events.csv", dtype=str)
    evmap = ev.set_index("source_path")
    for i in idxs:
        sid = f"val_{i:02d}"
        rows = comp[comp.stream_id == sid]
        segs = []
        h5cir = h5py.File(L.PKG / "streams" / "val" / f"stream_{i:02d}.h5", "r")["cir"]
        for r in rows.itertuples():
            kw = {}
            if r.kind == "fall":
                e = evmap.loc[r.path]
                kw = dict(event_t0=float(e.source_t_start), event_t1=float(e.source_t_end), fall_type=e.fall_type,
                          quality=e.quality, needs_review=e.needs_review == "true", ignore=e.ignore == "true")
            # The CSV keeps 3 decimals; a source time on an exact half frame (e.g. 0.350 s = 17.5 frames) is
            # ambiguous under np.round. Pick the +-1e-6 nudge whose block length equals the published one.
            n_pub = int(round((float(r.t_end) - float(r.t_start)) / L.STEP_S))
            o_pub = int(round(float(r.t_start) / L.STEP_S))
            t0, t1 = float(r.source_t0), float(r.source_t1)
            best = (t0, t1)
            for d0, d1 in [(0, 0), (-1e-6, -1e-6), (1e-6, 1e-6), (-1e-6, 0), (1e-6, 0), (0, -1e-6), (0, 1e-6),
                           (-1e-6, 1e-6), (1e-6, -1e-6)]:
                block, i0, i1 = L._slice(r.path, t0 + d0, t1 + d1)
                if i1 - i0 != n_pub:
                    continue
                # interior frames (outside both 0.2 s crossfades) must match the published stream
                inner = h5cir[0, o_pub + L.FADE_N : o_pub + n_pub - L.FADE_N]
                if np.array_equal(inner, block[0, L.FADE_N : n_pub - L.FADE_N]):
                    best = (t0 + d0, t1 + d1)
                    break
            segs.append(Segment(kind=r.kind, path=r.path, label=r.label if isinstance(r.label, str) else "",
                                dataset=r.dataset if isinstance(r.dataset, str) else "",
                                source_t0=best[0], source_t1=best[1], **kw))
        cir, events, parts = L.build_stream(sid, segs)
        with h5py.File(L.PKG / "streams" / "val" / f"stream_{i:02d}.h5", "r") as h:
            shape_ok = h["cir"].shape == cir.shape
            eq = [bool(np.array_equal(h["cir"][r], cir[r])) for r in range(3)] if shape_ok else [False]
        mine_ev = pd.DataFrame(events)[["t_start", "t_end", "source_path"]].reset_index(drop=True)
        pub_ev = ev[ev.stream_id == sid][["t_start", "t_end", "source_path"]].reset_index(drop=True)
        mine_co = pd.DataFrame(parts)[["t_start", "t_end"]].reset_index(drop=True)
        pub_co = rows[["t_start", "t_end"]].reset_index(drop=True)
        print(f"{sid}: shape_equal={shape_ok} cir_bit_equal_per_radar={eq} "
              f"events_equal={mine_ev.equals(pub_ev)} ({len(mine_ev)} ev) composition_times_equal={mine_co.equals(pub_co)}",
              flush=True)
        if "--no-extract" in sys.argv:
            continue
        # extractor parity on the real h5
        with h5py.File(L.PKG / "streams" / "val" / f"stream_{i:02d}.h5", "r") as h:
            F = L.extract(lambda r: h["cir"][r : r + 1])
        with np.load(FEATS / f"{sid}.npz") as z:
            res = {}
            for k in z.files:
                a, b = z[k], F[k]
                if a.shape != b.shape or a.dtype != b.dtype:
                    res[k] = f"shape/dtype {a.shape}{a.dtype} vs {b.shape}{b.dtype}"
                elif np.array_equal(a, b):
                    res[k] = "bit-equal"
                else:
                    rel = float(np.max(np.abs(a.astype(np.float64) - b) / (np.abs(a.astype(np.float64)) + 1e-12)))
                    res[k] = f"max_rel_diff={rel:.2e} n_diff={int((a != b).sum())}"
        print(f"{sid}: extractor vs cache: {res}", flush=True)
        del cir, F
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")


if __name__ == "__main__":
    main()

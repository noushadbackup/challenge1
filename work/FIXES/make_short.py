"""Build short single-stream fixtures (folder named `test`) by cropping a published val stream.

    python3 work/FIXES/make_short.py <out_root> <src_h5> <start_s> <len_s> [<len_s> ...]

Each length L gets <out_root>/L<L>/test/stream_04.h5 with the same datasets/attrs as the source
(cir, t, radar_serials). Used only to test that predict survives a short remainder stream on release day.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

root, src, start = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
with h5py.File(src, "r") as h:
    for L in map(float, sys.argv[4:]):
        i0, n = int(round(start * 50)), int(round(L * 50))
        d = root / f"L{L:g}" / "test"
        d.mkdir(parents=True, exist_ok=True)
        with h5py.File(d / "stream_04.h5", "w") as o:
            o["cir"] = np.asarray(h["cir"][:, i0:i0 + n], np.float32)
            o["t"] = np.asarray(h["t"][i0:i0 + n]) - h["t"][i0]
            o["radar_serials"] = h["radar_serials"][()]
            for k, v in h.attrs.items():
                o.attrs[k] = v
            o.attrs["split"] = "test"
        print(d / "stream_04.h5", n)

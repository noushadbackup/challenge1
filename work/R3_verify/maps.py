"""Range-time frame-difference energy maps from the raw CIR, one stream at a time (per-stream, no cross-stream info).

map[r, j, b] = log10(1e-12 + mean_{t in 0.1 s bin j} |cir_r[t, b] - cir_r[t-1, b]|^2)   for range bins b = 0..71
(complex bin = I[b] + j Q[b], I = cols 0..183, Q = cols 184..367). Stored float16 (3, T//5, 72) in
/home/user/data_raw/r3/verify/maps/<stream_id>.npy.

usage: python maps.py fold0 fold1 ... | official   (fold<k> = cv<k>_train_* + cv<k>_test_*; official = train_* + val_*)
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import srcv
from srcv import core as C

OUT = "/home/user/data_raw/r3/verify/maps"
NB = 72
BIN = 5


def stream_map(cir: np.ndarray) -> np.ndarray:
    T = cir.shape[1]
    n = T // BIN
    out = np.empty((3, n, NB), np.float16)
    for r in range(3):
        z = cir[r, :, :NB].astype(np.float32) + 1j * cir[r, :, 184:184 + NB].astype(np.float32)
        d = np.empty_like(z)
        d[0] = 0
        d[1:] = z[1:] - z[:-1]
        e = (d.real ** 2 + d.imag ** 2)[: n * BIN].reshape(n, BIN, NB).mean(1)
        out[r] = np.log10(1e-12 + e).astype(np.float16)
    return out


def sources_for(tag):
    if tag.startswith("fold"):
        k = int(tag[4:])
        return C.sources(k, "train") + C.sources(k, "test")
    if tag == "official":
        return C.real_sources("train") + C.real_sources("val")
    raise ValueError(tag)


def main():
    os.makedirs(OUT, exist_ok=True)
    for tag in sys.argv[1:]:
        for s in sources_for(tag):
            p = f"{OUT}/{s.stream_id}.npy"
            if os.path.exists(p):
                continue
            t0 = time.time()
            cir = C.load_cir(s)
            m = stream_map(cir)
            del cir
            np.save(p + ".tmp.npy", m)
            os.replace(p + ".tmp.npy", p)
            print(f"{s.stream_id} {m.shape} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

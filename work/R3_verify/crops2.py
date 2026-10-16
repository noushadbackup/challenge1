"""6-channel range-time crops straight from the raw CIR, one stream at a time (no full maps kept on disk).

Per radar r (0..2), 0.1 s bins, range bins 0..71 (z = I + jQ, I = cols 0..183, Q = cols 184..367):
  D_r = log10(1e-12 + mean |z_t - z_{t-1}|^2)   (frame-difference energy, = maps.py)
  A_r = log10(1e-12 + mean |z_t|)                (amplitude / static profile)
each minus the stream's own per-range-bin median. Crop = candidate centre +- 6 s (120 bins).
Channel order: D_0, D_1, D_2, A_0, A_1, A_2.  Out: /home/user/data_raw/r3/verify/crops2/<set>_{train,test}.npy
(n, 6, 72, 120) float16, rows in the order of work/R3_verify/cands/<set>_{train,test}.csv.

usage: python crops2.py fold0 ... official
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from srcv import core as C

OUT = "/home/user/data_raw/r3/verify/crops2"
CANDS = "/home/user/challenge1/work/R3_verify/cands"
NB, BIN, HALF = 72, 5, 60


def stream_maps(cir):
    T = cir.shape[1]
    n = T // BIN
    out = np.empty((6, n, NB), np.float32)
    for r in range(3):
        z = cir[r, :, :NB].astype(np.float32) + 1j * cir[r, :, 184:184 + NB].astype(np.float32)
        d = np.zeros_like(z)
        d[1:] = z[1:] - z[:-1]
        e = (d.real ** 2 + d.imag ** 2)[: n * BIN].reshape(n, BIN, NB).mean(1)
        a = np.abs(z)[: n * BIN].reshape(n, BIN, NB).mean(1)
        out[r] = np.log10(1e-12 + e)
        out[3 + r] = np.log10(1e-12 + a)
    out -= np.median(out, axis=1, keepdims=True)
    return out


def sources(S):
    if S.startswith("fold"):
        k = int(S[4:])
        return {s.stream_id: s for s in C.sources(k, "train") + C.sources(k, "test")}
    return {s.stream_id: s for s in C.real_sources("train") + C.real_sources("val")}


def main():
    os.makedirs(OUT, exist_ok=True)
    for S in sys.argv[1:]:
        if all(os.path.exists(f"{OUT}/{S}_{p}.npy") for p in ("train", "test")):
            continue
        t0 = time.time()
        srcs = sources(S)
        dfs = {p: pd.read_csv(f"{CANDS}/{S}_{p}.csv") for p in ("train", "test")}
        outs = {p: np.zeros((len(d), 6, NB, 2 * HALF), np.float16) for p, d in dfs.items()}
        sids = sorted(set(dfs["train"].stream_id) | set(dfs["test"].stream_id))
        for s in sids:
            cir = C.load_cir(srcs[s])
            m = stream_maps(cir)
            del cir
            n = m.shape[1]
            for p, d in dfs.items():
                for i in np.flatnonzero(d.stream_id.values == s):
                    c = int(round(0.5 * (d.t_start[i] + d.t_end[i]) * 10))
                    a, b = c - HALF, c + HALF
                    w = np.zeros((6, 2 * HALF, NB), np.float32)
                    aa, bb = max(a, 0), min(b, n)
                    w[:, aa - a: bb - a] = m[:, aa:bb]
                    outs[p][i] = np.transpose(w, (0, 2, 1)).astype(np.float16)
        for p in outs:
            np.save(f"{OUT}/{S}_{p}.npy", outs[p])
        print(f"{S}: {len(sids)} streams, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""E3: complex I/Q radar-physics features per 0.2 s block, from the raw h5 streams.

Usage: python3 extract_iq.py [n_jobs=2] [out_dir=/home/user/data_raw/feats_iq]

Reads one radar of one stream at a time (~130 MB float32), so peak RSS stays < 1 GB
per process. For each stream writes <out_dir>/<split>_<XX>.npz with arrays of shape
(3, n_blocks) (n_blocks = T // 10, same grid as prof_02 in the cached features).

Physics choices (data-team verdict F1, F5, repeat finding; expert X1 P1):
  * z = I + jQ with I = values 0..183, Q = values 184..367.
  * Only the mat range zone of each radar: R29171 (idx 0) bins 40..68 (D3 far bins),
    R29444 (idx 1) and R30722 (idx 2) bins 5..21.
  * Bit-exact repeated frames (sample-and-hold, ~25 %) are dropped; all lag-1 quantities
    are computed between consecutive DISTINCT frames with the true gap (dt = gap * 20 ms).
  * Clutter removal = centred moving mean over 25 distinct frames (~0.6 s) on the distinct
    sequence (short, so a junction only pollutes ~+-0.3 s; no 3 s EMA).
  * No phase unwrapping / integration: only wrap-robust lag-1 statistics.

Stored per radar and block (sums, so they can be re-aggregated at any scale):
  cnt      number of distinct frames in the block
  P_re/P_im, E0, E1    raw lag-1 complex correlation sum_b z_k conj(z_{k-1}), energies
  Q_re/Q_im            high-passed lag-1 correlation sum_b zh_k conj(zh_{k-1})
  Sw, Swhi             sum_b |zh_k zh_{k-1}|, and the part with |dphi| > pi/2 (aliased / fast)
  Eh                   high-passed energy sum_b |zh_k|^2
  Qdt_re/Qdt_im        sum of (Q / |Q|) per pair (unit phasors), and sdt = sum dt -> mean Doppler
  Md, Mz               rate-normalised diff energy (all 368 values / zone only): sum (x_k-x_{k-1})^2 / gap
  Cb0, Cb1, Cb2        sum_b E_b, sum_b b E_b, sum_b b^2 E_b of high-passed energy (range centroid/spread)
"""
from __future__ import annotations

import sys
import time
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
ZONES = {0: (40, 69), 1: (5, 22), 2: (5, 22)}
HP_WIN = 25
KEYS = ["cnt", "P_re", "P_im", "E0", "E1", "Q_re", "Q_im", "Sw", "Swhi", "Eh", "U_re", "U_im", "sdt", "Md", "Mz",
        "Cb0", "Cb1", "Cb2"]


def radar_blocks(x: np.ndarray, zone: tuple[int, int]) -> dict[str, np.ndarray]:
    T = x.shape[0]
    n = T // 10
    d = np.ones(T, bool)
    d[1:] = np.any(x[1:] != x[:-1], axis=1)
    idx = np.flatnonzero(d)
    a, b = zone
    zd = (x[idx, a:b] + 1j * x[idx, 184 + a:184 + b]).astype(np.complex64)  # (K, B)
    gap = np.diff(idx).astype(np.float32)  # pairs k=1..K-1
    blk = idx[1:] // 10
    keep = blk < n
    out: dict[str, np.ndarray] = {}

    def agg(name, w):
        out[name] = np.bincount(blk[keep], weights=np.asarray(w, np.float64)[keep], minlength=n)[:n].astype(np.float32)

    out["cnt"] = np.bincount(idx[idx // 10 < n] // 10, minlength=n)[:n].astype(np.float32)
    # raw lag-1 coherence (static clutter dominated)
    P = np.sum(zd[1:] * np.conj(zd[:-1]), axis=1)
    e = np.sum(np.abs(zd) ** 2, axis=1)
    agg("P_re", P.real), agg("P_im", P.imag), agg("E0", e[:-1]), agg("E1", e[1:])
    # high-pass on the distinct sequence
    zh = zd - (uniform_filter1d(zd.real, HP_WIN, axis=0, mode="nearest")
               + 1j * uniform_filter1d(zd.imag, HP_WIN, axis=0, mode="nearest")).astype(np.complex64)
    Qb = zh[1:] * np.conj(zh[:-1])  # (K-1, B)
    w = np.abs(Qb)
    hi = np.abs(np.angle(Qb)) > np.pi / 2
    Q = Qb.sum(1)
    agg("Q_re", Q.real), agg("Q_im", Q.imag)
    agg("Sw", w.sum(1)), agg("Swhi", (w * hi).sum(1))
    eh = np.abs(zh[1:]) ** 2  # (K-1, B)
    agg("Eh", eh.sum(1))
    u = Q / (np.abs(Q) + 1e-30)
    agg("U_re", u.real), agg("U_im", u.imag), agg("sdt", gap * 0.02)
    # rate-normalised frame-difference motion (spreads the jump of a hold over its length)
    dx = x[idx[1:]] - x[idx[:-1]]
    agg("Md", np.einsum("ij,ij->i", dx, dx) / gap)
    agg("Mz", np.sum(np.abs(zd[1:] - zd[:-1]) ** 2, axis=1) / gap)
    bins = np.arange(a, b, dtype=np.float64)
    agg("Cb0", eh.sum(1)), agg("Cb1", eh @ bins), agg("Cb2", eh @ bins**2)
    return out


def process(args: tuple[str, str]) -> str:
    path, out = Path(args[0]), Path(args[1])
    if out.exists():
        return f"skip {out.name}"
    t0 = time.time()
    res = {k: [] for k in KEYS}
    with h5py.File(path, "r") as h:
        for r in range(3):
            x = h["cir"][r]  # (T, 368) float32
            o = radar_blocks(x, ZONES[r])
            del x
            for k in KEYS:
                res[k].append(o[k])
    np.savez_compressed(out, **{k: np.stack(v) for k, v in res.items()})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return f"done {out.name} {time.time() - t0:.0f}s maxrss={rss:.2f}GB"


def main() -> None:
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/user/data_raw/feats_iq")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(out_dir / f"{p.parent.name}_{p.stem.split('_')[1]}.npz"))
             for p in sorted((PKG / "streams").glob("*/stream_*.h5"))]
    with ProcessPoolExecutor(jobs) as pool:
        for msg in pool.map(process, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

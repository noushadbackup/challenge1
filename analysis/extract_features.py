"""Compress every stream into compact per-frame / per-block features for analysis.

Usage: python analysis/extract_features.py <package_root> <out_dir> [n_jobs]

For each stream_XX.h5 writes <out_dir>/<split>_<XX>.npz containing
  e_raw    (3, T)          raw frame energy  sum_bins x^2
  e_mot    (3, T)          residual energy after 3 s causal EMA background removal
  e_diff   (3, T)          frame-to-frame difference energy  sum_bins (x_t - x_{t-1})^2
  mot_map  (3, T//10, 184) mean |complex residual| per range bin, 0.2 s blocks (float16)
  amp_1s   (3, T//50, 184) |mean complex CIR| per range bin, 1 s blocks -> static amplitude profile
  prof_1s  (3, T//50, 368) mean raw CIR per value, 1 s blocks (float32) -> static fingerprint
  prof_02  (3, T//10, 368) 1000 x mean raw CIR per value, 0.2 s blocks (float16) -> fine static profile

The 368 values of a frame are treated as 184 range bins of I (0..183) followed by
Q (184..367); see reports/REPORT.md section 2 for the evidence.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import lfilter, lfilter_zi

RATE = 50.0
TAU = 3.0


def ema_residual(cir: np.ndarray, tau: float = TAU) -> np.ndarray:
    alpha = np.exp(-1.0 / (RATE * tau))
    b, a = [1.0 - alpha], [1.0, -alpha]
    zi = lfilter_zi(b, a)[0] * cir[:, :1, :]
    avg, _ = lfilter(b, a, cir, axis=1, zi=zi)
    return (cir - avg).astype(np.float32)


def to_complex(cir: np.ndarray) -> np.ndarray:
    return cir[..., :184] + 1j * cir[..., 184:]


def block_mean(x: np.ndarray, size: int) -> np.ndarray:
    n = x.shape[1] // size
    return x[:, : n * size].reshape(x.shape[0], n, size, *x.shape[2:]).mean(axis=2)


def process(args: tuple[str, str]) -> str:
    path, out = Path(args[0]), Path(args[1])
    if out.exists():
        return f"skip {out.name}"
    with h5py.File(path, "r") as h:
        cir = h["cir"][:].astype(np.float64)
    res = ema_residual(cir)
    diff = np.diff(cir, axis=1, prepend=cir[:, :1])
    np.savez_compressed(
        out,
        e_raw=(cir**2).sum(2).astype(np.float32),
        e_mot=(res.astype(np.float64) ** 2).sum(2).astype(np.float32),
        e_diff=(diff**2).sum(2).astype(np.float32),
        mot_map=block_mean(np.hypot(res[..., :184], res[..., 184:]), 10).astype(np.float16),
        amp_1s=np.abs(block_mean(to_complex(cir), 50)).astype(np.float32),
        prof_1s=block_mean(cir, 50).astype(np.float32),
        prof_02=(1e3 * block_mean(cir, 10)).astype(np.float16),
    )
    return f"done {out.name} T={cir.shape[1]}"


def main() -> None:
    root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(p), str(out_dir / f"{p.parent.name}_{p.stem.split('_')[1]}.npz"))
        for p in sorted((root / "streams").glob("*/stream_*.h5"))
    ]
    with ProcessPoolExecutor(jobs) as pool:
        for msg in pool.map(process, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

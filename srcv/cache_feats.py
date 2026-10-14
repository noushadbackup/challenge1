"""Pre-compute the feature cache (E1/analysis extractor keys) for SR-CV virtual streams.

    python3 -m srcv.cache_feats [--folds 0,1,2,3] [--parts test,train] [--workers 2]

Each stream is rebuilt in memory (bit-exact writer replica) and only the npz is written to
/home/user/data_raw/srcv/feats/<stream_id>.npz (+ .json sidecar with the plan hash). ~1.5 GB RSS per worker.
"""
from __future__ import annotations

import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from . import core as C


def _one(src) -> tuple[str, float, float]:
    t0 = time.time()
    C.cache_feats(src)
    return src.stream_id, time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> None:
    a = sys.argv[1:]
    folds = [int(x) for x in a[a.index("--folds") + 1].split(",")] if "--folds" in a else [0, 1, 2, 3]
    parts = a[a.index("--parts") + 1].split(",") if "--parts" in a else ["test", "train"]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 2
    todo = [s for part in parts for k in folds for s in C.sources(k, part)
            if not (C.FEATS / f"{s.stream_id}.npz").exists()]
    print(f"{len(todo)} streams to extract", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for sid, dt, rss in pool.map(_one, todo):
            print(f"{sid}: {dt:.0f}s worker peak RSS {rss:.2f} GB", flush=True)


if __name__ == "__main__":
    main()

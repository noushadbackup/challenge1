"""Generate the pseudo-test bank from the VALIDATION segments with new seeds.

Planning: abc2027.streams.prepare_split(split='test', rng=default_rng(seed)) -> test settings
(adl_uses=2, flagged falls get ignore=true) + finish_plan(density=57.1864, configs/streams.yaml
target, which yields ~58 falls/h after crossfades, as in the published streams).
Writing: e1lib.build_stream (bit-exact replica of build_streams._write_stream, verified by
check_parity.py), then e1lib.extract (bit-exact replica of analysis/extract_features.py).
Only features (npz) + events/composition CSVs are stored; no h5.

Usage: python3 gen_bank.py <seed> [<seed> ...]   (each seed = one full pseudo-val split, ~5 streams)
Output: /home/user/data_raw/pseudo/feats/p<seed>_<XX>.npz, events_<seed>.csv, composition_<seed>.csv
"""
from __future__ import annotations

import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import e1lib as L  # noqa: I001
import numpy as np
from abc2027.streams import finish_plan, prepare_split

DENSITY = 57.1864  # configs/streams.yaml target_falls_per_hour (binding density of the public plan)
FEATS = L.OUT / "feats"


def work(task):
    sid, segments = task
    t0 = time.time()
    out = FEATS / f"{sid}.npz"
    cir, events, parts = L.build_stream(sid, segments)
    if not out.exists():
        F = L.extract(lambda r: cir[r : r + 1])
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **F)
        tmp.rename(out)
    T = cir.shape[1]
    del cir
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return sid, events, parts, T, time.time() - t0, rss


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]]
    FEATS.mkdir(parents=True, exist_ok=True)
    group = L.val_groups()
    print(f"val pool: falls={len(group['falls'])} adl_files={len(group['adl'])} others={len(group['others'])}", flush=True)
    for seed in seeds:
        if (L.OUT / f"events_{seed}.csv").exists():
            print(f"seed {seed}: exists, skip", flush=True)
            continue
        bundle = prepare_split("test", group["falls"], group["adl"], group["others"], rng=np.random.default_rng(seed))
        plan = finish_plan(bundle, DENSITY)
        tasks = [(f"p{seed}_{i:02d}", st.segments) for i, st in enumerate(plan.streams)]
        all_ev, all_co, meta = [], [], []
        with ProcessPoolExecutor(2) as pool:
            for sid, events, parts, T, dt, rss in pool.map(work, tasks):
                all_ev += events
                all_co += parts
                meta.append({"stream_id": sid, "T": T, "dur_s": T / 50.0, "n_events": len(events), "sec": round(dt, 1),
                             "worker_peak_rss_gb": round(rss, 2)})
                print(f"seed {seed} {sid}: T={T} events={len(events)} {dt:.0f}s rss={rss:.2f}GB", flush=True)
        for row in all_ev:
            row["seed"] = seed
        for row in all_co:
            row["seed"] = seed
        L.write_csv(L.OUT / f"composition_{seed}.csv", all_co)
        L.write_csv(L.OUT / f"events_{seed}.csv", all_ev)
        (L.OUT / f"meta_{seed}.json").write_text(json.dumps(
            {"seed": seed, "density_arg": DENSITY, "plan_density": plan.density_per_hour, "streams": meta}, indent=1))


if __name__ == "__main__":
    main()

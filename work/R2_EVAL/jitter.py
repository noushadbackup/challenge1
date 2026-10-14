"""Extra replay crops at jittered left offsets for the C3/C4 candidates (mechanism: replay result depends
chaotically on where the crop starts, i.e. on the detector start; more crop placements -> more stable median).

    python3 work/R2_EVAL/jitter.py <bag> [bases C3,C4] [targets]
Reads the candidates of cache/D_<bag>_<t>_<base>.csv, replays the extra crops JCROPS on the same candidates and
writes cache/DJ_<bag>_<t>_<base>.csv (same row order). variants.py then forms 'J' (B's 6 crops + 6 jittered).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import uwbfall
from uwbfall.boundary import replay_stream
from uwbfall.config import load_config
from uwbfall.extract import get_features
from uwbfall.models import setup_torch
from srcv import bridge, core as C

HERE = Path(__file__).resolve().parent
JCROPS = ["loc1.5_4d", "loc2.5_4d", "det1.5_4d", "det2.5_4d", "detq1.5_4d", "detq2.5_4d"]


def main():
    bag = sys.argv[1]
    bases = sys.argv[2].split(",") if len(sys.argv) > 2 else ["C3", "C4"]
    targets = sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1", "2", "3", "official"]
    cfg = load_config({"boundary": {"crops": JCROPS}})
    _lim = setup_torch(cfg)  # noqa: F841
    for t in targets:
        te = uwbfall.split_sources("val") if t == "official" else bridge.pipeline_sources(int(t), "test")
        Ds = {b: pd.read_csv(HERE / "cache" / f"D_{bag}_{t}_{b}.csv", float_precision="round_trip") for b in bases}
        outs = {b: [] for b in bases}
        for s in sorted(te, key=lambda s: s.sid):
            todo = {b: D[D.stream_id == s.sid][["stream_id", "t_start", "t_end", "score"]] for b, D in Ds.items()}
            if not any(len(v) for v in todo.values()):
                continue
            F = get_features(s, cfg)
            for b, c in todo.items():
                if len(c):
                    outs[b].append(replay_stream(s, F["prof_02"], c.reset_index(drop=True), cfg))
            s.release()
        for b in bases:
            DJ = pd.concat(outs[b], ignore_index=True)
            assert len(DJ) == len(Ds[b]) and (DJ.t_start.values == Ds[b].t_start.values).all()
            DJ.to_csv(HERE / "cache" / f"DJ_{bag}_{t}_{b}.csv", index=False)
        print(f"[{bag}] jitter target {t} done", flush=True)


if __name__ == "__main__":
    main()

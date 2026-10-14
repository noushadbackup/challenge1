"""Virtual-source writer parity: rebuild published streams from streams_composition.csv (3-decimal rows,
half-frame ambiguity resolved from published stream times only) and compare bit for bit with the h5.
Usage: PYTHONPATH=/home/user/challenge1 python3 -m uwbfall.tests.parity_virtual [val|train] [idx ...]
"""
import sys

import numpy as np
import pandas as pd

from uwbfall.paths import PKG
from uwbfall.sources import sources_from_composition, split_sources

split = sys.argv[1] if len(sys.argv) > 1 else "val"
idx = [int(a) for a in sys.argv[2:]] or None
comp = pd.read_csv(PKG / "streams_composition.csv", dtype=str)
ev = pd.read_csv(PKG / "streams_events.csv", dtype=str)
ok = True
for h in split_sources(split):
    if idx is not None and int(h.sid.split("_")[1]) not in idx:
        continue
    v = sources_from_composition(comp, ev, [h.sid])[0]
    same_len = v.n_frames == h.n_frames
    eq = [same_len and bool(np.array_equal(v.radar(r)[0], h.radar(r)[0])) for r in range(3)]
    e = v.events()[["t_start", "t_end", "source_path"]].reset_index(drop=True)
    pe = ev[ev.stream_id == h.sid][["t_start", "t_end", "source_path"]].reset_index(drop=True)
    c = v.composition()[["t_start", "t_end"]].reset_index(drop=True)
    pc = comp[comp.stream_id == h.sid][["t_start", "t_end"]].reset_index(drop=True)
    r = dict(n_frames=(v.n_frames, h.n_frames), cir_bit_equal=eq, events_equal=e.equals(pe), comp_times_equal=c.equals(pc))
    ok &= all(eq) and r["events_equal"] and r["comp_times_equal"]
    print(h.sid, r, flush=True)
    v.release(); h.release()
print("VIRTUAL PARITY OK" if ok else "VIRTUAL PARITY FAILED")
sys.exit(0 if ok else 1)

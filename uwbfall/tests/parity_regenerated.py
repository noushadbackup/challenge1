"""Virtual sources from a regenerated plan (full-precision Segment objects, E1 pseudo-bank seed) ->
uwbfall extractor -> compare with E1's stored bank features /home/user/data_raw/pseudo/feats/p<seed>_<XX>.npz.
Uses work/E1_pseudo_test/e1lib.val_groups only to rebuild the planning input (test-only import).
Usage: PYTHONPATH=/home/user/challenge1 python3 -m uwbfall.tests.parity_regenerated [seed] [stream_idx ...]
"""
import sys

import numpy as np

sys.path.insert(0, "/home/user/challenge1/work/E1_pseudo_test")
import e1lib  # noqa: E402
from abc2027.streams import finish_plan, prepare_split  # noqa: E402

from uwbfall.extract import extract  # noqa: E402
from uwbfall.sources import VirtualSource  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1001
idx = [int(a) for a in sys.argv[2:]] or [4]
g = e1lib.val_groups()
plan = finish_plan(prepare_split("test", g["falls"], g["adl"], g["others"], rng=np.random.default_rng(seed)), 57.1864)
ok = True
for i in idx:
    v = VirtualSource(f"p{seed}_{i:02d}", plan.streams[i].segments)
    F = extract(v)
    with np.load(f"/home/user/data_raw/pseudo/feats/p{seed}_{i:02d}.npz") as z:
        r = {k: bool(np.array_equal(F[k], z[k])) for k in F}
    ok &= all(r.values())
    print(v.sid, "T", v.n_frames, r, "events", len(v.events()), flush=True)
    v.release()
print("REGENERATED PARITY OK" if ok else "REGENERATED PARITY FAILED")
sys.exit(0 if ok else 1)

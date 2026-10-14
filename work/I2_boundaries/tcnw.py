"""(c) TCN arm: TCN-only bag fitted with labels.ignore_clips = zero_weight (whole ignore-zone fall clips get zero
BCE weight), same seeds as the bag named by --seed-offset (0 = uwb_default/E4 seeds). Then caches test
probabilities like R2_EVAL (cache/probs_<name>_<t>.npz with '<sid>|tcn').

    python3 work/I2_boundaries/tcnw.py <name> <seed_offset> <targets comma>
Bundles -> /home/user/data_raw/I2/runs/<name>/<target dir> (outside git)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, target_sources, tdir  # noqa: E402
import uwbfall  # noqa: E402
from uwbfall import pipeline as PL  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from srcv import bridge  # noqa: E402

name, off = sys.argv[1], int(sys.argv[2])
ROOT = Path("/home/user/data_raw/I2/runs") / name
cfg = {"hgb": {"enabled": False}, "tcn": {"seed_offset": off}, "report": {"oof_replay": False},
       "labels": {"ignore_clips": "zero_weight", "ignore_members": ["tcn"]},
       "clip": {"segclf": {"enabled": False}, "gate": "none"}, "decode": {"one_event_per_clip": False}}
for t in sys.argv[3].split(","):
    t0 = time.time()
    out = ROOT / tdir(t)
    tr, ev, te, _ = target_sources(t)
    if not (out / "meta.json").exists():
        if t != "official":
            bridge.seed_uwbfall_cache(int(t), "train", uwbfall.load_config(cfg))
        uwbfall.fit(tr, ev, cfg, out_dir=out)
    B = PL.load_bundle(out)
    P = {}
    for s in te:
        _p, parts = PL.stream_probs(B, get_features(s, B["cfg"]))
        P[f"{s.sid}|tcn"] = parts["tcn"]
        s.release()
    np.savez_compressed(HERE / "cache" / f"probs_{name}_{t}.npz", oof=np.load(out / "oof.npz")["oof_tcn"], **P)
    print(f"[{name}] {t} done in {time.time() - t0:.0f}s", flush=True)

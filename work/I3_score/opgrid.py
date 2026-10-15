"""I3 (b) operating point on SR-CV OOF: replay internals for EVERY decode-grid candidate of the test parts.

For each target (SR-CV folds 0-3 and official val) and bag: G7-gated blend test probabilities, every point of the
uwbfall decode grid (thr x smooth x min_dur x gap = 60), candidates -> union over grid and bags -> replay the rows
not yet cached (reuses I2's and run3's replay caches, read-only) -> cache/repG_<t>_test.pkl.
Then opsel.py selects the operating point on the OTHER folds' final (replay + NMS) ranking F1.

    python3 work/I3_score/opgrid.py <targets> [--count-only]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
from common import BAGS, REF_CFG, Probs, target_sources  # noqa: E402
from uwbfall.boundary import replay_stream  # noqa: E402
from uwbfall.decode import events_df, grid, to_events  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from uwbfall.models import setup_torch  # noqa: E402

KEY = ["stream_id", "t_start", "t_end"]
I2C = HERE.parent / "I2_boundaries" / "cache"


def cached(t):
    parts = [pickle.load(open(I2C / f"rep_{t}_test.pkl", "rb"))["D"]]
    for f in (HERE / "cache" / f"repR_{t}_test.pkl", HERE / "cache" / f"repG_{t}_test.pkl"):
        if f.exists():
            parts.append(pickle.load(open(f, "rb")))
    return pd.concat(parts, ignore_index=True).drop_duplicates(KEY).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    cfg = REF_CFG
    _lim = setup_torch(cfg)  # noqa: F841
    G = grid(cfg)
    for t in a.targets.split(","):
        t0 = time.time()
        _tr, _e, te, _ = target_sources(t)
        te = sorted(te, key=lambda s: s.sid)
        D = cached(t)
        allc = []
        for bag in BAGS:
            P = Probs(bag, t)
            for s in te:
                p = P.test(s.sid)
                for g in G:
                    allc += to_events(s.sid, p, *g)
        allc = events_df(allc).drop_duplicates(KEY)
        miss = allc.merge(D[KEY], on=KEY, how="left", indicator=True)
        miss = miss[miss._merge == "left_only"].drop(columns="_merge")
        print(t, "grid union", len(allc), "missing replays", len(miss), flush=True)
        if a.count_only:
            continue
        new = []
        for s in te:
            c = miss[miss.stream_id == s.sid].sort_values(KEY).reset_index(drop=True)
            if len(c):
                new.append(replay_stream(s, get_features(s, cfg)["prof_02"], c, cfg).drop(columns=["score"]))
                s.release()
        D = pd.concat([D] + new, ignore_index=True) if new else D
        pickle.dump(D, open(HERE / "cache" / f"repG_{t}_test.pkl", "wb"))
        print(f"{t}: {len(miss)} new replays ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

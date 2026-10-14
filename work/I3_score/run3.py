"""I3 version R (two-stage): recall-oriented candidate generator + hit-model filter.

Candidates: each bag's G7 operating point with thr -> 0.2 and min_dur -> 0.4 (pre-declared, not tuned), on the
G7-gated blend of the test parts of every target (SR-CV folds and official val). Replay internals: reuse the I2
union cache (work/I2_boundaries/cache/rep_<t>_test.pkl, read-only) and replay only the missing rows (cached in
cache/repR_<t>_test.pkl). Writes cache/featR_<bag>_<t>.pkl in run.py's format, consumed by
`run2.py --feat featR`.

    python3 work/I3_score/run3.py <targets> [--thr 0.2 --min-dur 0.4]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
from common import BAGS, REF_CFG, Probs, ops_g7, target_sources  # noqa: E402
from uwbfall import hitmodel as HM  # noqa: E402
from uwbfall.boundary import replay_stream  # noqa: E402
from uwbfall.decode import events_df, to_events  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from uwbfall.models import setup_torch  # noqa: E402

KEY = ["stream_id", "t_start", "t_end"]
I2C = HERE.parent / "I2_boundaries" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--min-dur", type=float, default=0.4)
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    cfg = REF_CFG
    b = cfg["boundary"]
    _lim = setup_torch(cfg)  # noqa: F841
    prep_peaks = {}
    for t in a.targets.split(","):
        t0 = time.time()
        _tr, _e, te, _ = target_sources(t)
        te = sorted(te, key=lambda s: s.sid)
        dur = {s.sid: s.duration for s in te}
        prep_peaks = pickle.load(open(HERE.parent / "I1_clip" / "cache" / f"prep_{t}.pkl", "rb"))["peaks"]
        rf = HERE / "cache" / f"repR_{t}_test.pkl"
        D = pickle.load(open(rf, "rb")) if rf.exists() else pickle.load(open(I2C / f"rep_{t}_test.pkl", "rb"))["D"]
        P, C = {}, {}
        for bag in BAGS:
            P[bag] = Probs(bag, t)
            op = list(ops_g7(bag, t))
            rop = (min(op[0], a.thr), op[1], min(op[2], a.min_dur), op[3])
            C[bag] = events_df(sum((to_events(s.sid, P[bag].test(s.sid), *rop) for s in te), []))
        allc = pd.concat(C.values()).drop_duplicates(KEY)
        miss = allc.merge(D[KEY], on=KEY, how="left", indicator=True)
        miss = miss[miss._merge == "left_only"].drop(columns="_merge")
        print(t, {bg: len(c) for bg, c in C.items()}, "union", len(allc), "missing replays", len(miss), flush=True)
        if a.count_only:
            continue
        new = []
        for s in te:
            c = miss[miss.stream_id == s.sid].sort_values(KEY).reset_index(drop=True)
            if len(c):
                new.append(replay_stream(s, get_features(s, cfg)["prof_02"], c, cfg).drop(columns=["score"]))
                s.release()
        if new:
            D = pd.concat([D] + new, ignore_index=True)
        pickle.dump(D, open(rf, "wb"))
        for bag in BAGS:
            M = C[bag].merge(D, on=KEY, how="left")
            assert M[f"{b['crops'][0]}_s"].notna().any()
            p = {s.sid: P[bag].test(s.sid) for s in te}
            parts = {s.sid: {"tcn": P[bag].Pt[f"{s.sid}|tcn"], "hgb": P[bag].Pt[f"{s.sid}|hgb"],
                             "gate": P[bag].gates[s.sid]} for s in te}
            X = HM.candidate_features(M, b, p, parts, prep_peaks)
            rows = HM.final_rows(M, b, dur)
            pickle.dump({"Xte": X.iloc[rows.row.values].reset_index(drop=True), "rows_te": rows},
                        open(HERE / "cache" / f"featR_{bag}_{t}.pkl", "wb"))
        print(f"{t}: {len(miss)} new replays ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

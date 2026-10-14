"""Replay internals of the reference (G7) candidates, train OOF and test, for all 3 bags of a target.

Candidates: G7-gated 0.75 TCN + 0.25 HGB at each bag's G7 operating point (I1 ops). The union of candidate
intervals of the 3 bags is replayed once per stream (replay of a row depends only on its interval), then split
back per bag. Test rows -> refine (reference gate 0.3) -> time NMS 4 s must equal I1 preds/<bag>/G7N4 (checked).

    python3 work/I2_boundaries/dump.py <targets comma> [--parts train,test]
Output: cache/rep_<t>_<part>.pkl = {"D": union replay table, "cand": {bag: candidates}, "ops": {bag: op}}
"""
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import BAGS, COLS, HERE, I1, REF_CFG, Probs, oname, ops_g7, target_sources  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall.boundary import refine, replay_stream, to_preds  # noqa: E402
from uwbfall.decode import decode_streams, events_df, to_events  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from uwbfall.models import setup_torch  # noqa: E402


def replay_union(sources, cands: dict, cfg):
    allc = pd.concat(cands.values(), ignore_index=True)
    allc = allc.drop_duplicates(["stream_id", "t_start", "t_end"]).sort_values(["stream_id", "t_start", "t_end"])
    out = []
    for s in sorted(sources, key=lambda s: s.sid):
        c = allc[allc.stream_id == s.sid].reset_index(drop=True)
        if not len(c):
            continue
        F = get_features(s, cfg)
        out.append(replay_stream(s, F["prof_02"], c, cfg))
        s.release()
    return pd.concat(out, ignore_index=True).drop(columns=["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("--parts", default="test,train")
    a = ap.parse_args()
    cfg = REF_CFG
    _lim = setup_torch(cfg)  # noqa: F841
    for t in a.targets.split(","):
        tr, _evtr, te, _evte = target_sources(t)
        P = {b: Probs(b, t) for b in BAGS}
        ops = {b: ops_g7(b, t) for b in BAGS}
        for part in a.parts.split(","):
            t0 = time.time()
            srcs = te if part == "test" else tr
            cands = {}
            for b in BAGS:
                if part == "test":
                    cands[b] = events_df(sum((to_events(s.sid, P[b].test(s.sid), *ops[b]) for s in srcs), []))
                else:
                    cands[b] = decode_streams(P[b].oof(), ops[b])
            D = replay_union(srcs, cands, cfg)
            pickle.dump({"D": D, "cand": cands, "ops": ops}, open(HERE / "cache" / f"rep_{t}_{part}.pkl", "wb"))
            msg = f"{t} {part}: {len(D)} replayed rows ({time.time() - t0:.0f}s)"
            if part == "test":
                dur = {s.sid: s.duration for s in srcs}
                for b in BAGS:
                    M = cands[b].merge(D, on=["stream_id", "t_start", "t_end"], how="left")
                    st, en, _ = refine(M, cfg["boundary"])
                    pr = CL.nms_time(to_preds(M, st, en, dur), 4.0).reset_index(drop=True)[COLS]
                    ref = pd.read_csv(I1 / "preds" / b / "G7N4" / f"{oname(t)}.csv")
                    pr2 = pd.read_csv(__import__("io").StringIO(pr.to_csv(index=False)))
                    ok = pr2.equals(ref) or (len(pr2) == len(ref) and np.allclose(pr2[COLS[1:]].values, ref[COLS[1:]].values)
                                             and (pr2.stream_id.values == ref.stream_id.values).all())
                    msg += f" | {b} G7N4 match={ok}"
            print(msg, flush=True)


if __name__ == "__main__":
    main()

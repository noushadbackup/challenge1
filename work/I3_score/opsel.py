"""I3 (b) operating point chosen on SR-CV OOF with nested (leave-target-fold-out) selection.

For each bag and every decode-grid point g: final rows (G7-gated blend -> candidates at g -> replay refine ->
time NMS 4 s) of each SR-CV test part and of val (replays from opgrid.py). For target fold k the operating point
maximises the pooled official ranking F1 over the 3 OTHER folds (first maximum in grid order, like
uwbfall.decode.tune); for val all 4 folds are pooled. Two criteria:
  XOPd  detector-level F1 of the raw candidates (exactly tune()'s criterion, but on SR-CV OOF instead of train OOF)
  XOPf  final-level F1 (after replay + NMS: what is submitted)
Sanity: at the bag's reference G7 op the final rows equal REF byte-for-byte (asserted).

    python3 work/I3_score/opsel.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
from common import BAGS, COLS, REF_CFG, Probs, oname, ops_g7, target_sources  # noqa: E402
from run import nms, write  # noqa: E402
from uwbfall import hitmodel as HM  # noqa: E402
from uwbfall.decode import events_df, grid, to_events  # noqa: E402
from uwbfall.scoring import gt_records, score_inproc  # noqa: E402
from uwbfall.sources import events_for  # noqa: E402

KEY = ["stream_id", "t_start", "t_end"]
TARGETS = ["0", "1", "2", "3", "official"]
FOLDS = TARGETS[:4]


def rf1(df, gt, dur):
    return score_inproc(df, gt, dur)["ranking_score"]["f1_mean"]


def main():
    b = REF_CFG["boundary"]
    G = grid(REF_CFG)
    gts, durs, srcs = {}, {}, {}
    for t in TARGETS:
        _tr, _e, te, evte = target_sources(t)
        srcs[t] = sorted(te, key=lambda s: s.sid)
        gts[t] = gt_records(events_for(te, evte), ignore_flagged=True)
        durs[t] = {s.sid: s.duration for s in te}
    D = {t: pickle.load(open(HERE / "cache" / f"repG_{t}_test.pkl", "rb")) for t in TARGETS}
    info = {}
    for bag in BAGS:
        cand, final = {}, {}
        for t in TARGETS:
            P = Probs(bag, t)
            pr = {s.sid: P.test(s.sid) for s in srcs[t]}
            for gi, g in enumerate(G):
                c = events_df(sum((to_events(sid, p, *g) for sid, p in pr.items()), []))
                cand[t, gi] = c
                if len(c):
                    M = c.merge(D[t], on=KEY, how="left")
                    assert M[f"{b['crops'][0]}_s"].notna().sum() + 0 >= 0 and len(M) == len(c)
                    final[t, gi] = nms(HM.final_rows(M, b, durs[t]).drop(columns="row"))[COLS]
                else:
                    final[t, gi] = c
            gref = G.index(tuple(ops_g7(bag, t)))
            ref = pd.read_csv(HERE / "preds" / bag / "REF" / f"{oname(t)}.csv")
            mine = pd.read_csv(__import__("io").StringIO(final[t, gref].to_csv(index=False)))
            assert mine.equals(ref), (bag, t)
        # per-fold scores for every grid point, then nested choice
        rec = {}
        for t in TARGETS:
            train = [f for f in FOLDS if f != t]
            gt = sum((gts[f] for f in train), [])
            dur = {k: v for f in train for k, v in durs[f].items()}
            sd = [rf1(pd.concat([cand[f, gi] for f in train]), gt, dur) for gi in range(len(G))]
            sf = [rf1(pd.concat([final[f, gi] for f in train]), gt, dur) for gi in range(len(G))]
            jd, jf = int(np.argmax(sd)), int(np.argmax(sf))
            write(bag, "XOPd", t, final[t, jd])
            write(bag, "XOPf", t, final[t, jf])
            rec[t] = {"ref_op": list(ops_g7(bag, t)), "op_d": list(G[jd]), "op_f": list(G[jf]),
                      "srcv_train_f1_ref_op": sf[G.index(tuple(ops_g7(bag, t)))], "srcv_train_f1_op_f": sf[jf],
                      "srcv_train_detF1_op_d": sd[jd]}
            print(bag, t, rec[t], flush=True)
        info[bag] = rec
    (HERE / "out" / "opsel.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

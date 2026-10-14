"""I3 version X: hit model and threshold learned on SR-CV OOF candidates (the held-out test parts of the OTHER
SR-CV folds), as PLAN W5 asks ("on SR-CV OOF candidate events"). The 4 SR-CV test parts are pairwise disjoint in
fall subjects and ADL pieces (checked in the report), so for target fold k nothing of fold k's subjects/pieces
enters the training rows. For the official val target (and the final test submission) all 4 folds are used.

Nesting: tau is chosen on the training folds with hit scores from leave-one-training-fold-out hit models, i.e.
scores of rows never seen by the model that scored them; the final hit model for the target uses all training folds.

Inputs: work/I3_score/cache/feat_<bag>_<t>.pkl from run.py (test-part features/rows of every target).
Variants: XT0 (tau on detector score), XH (hit-model score), XHT (hit score + nested tau). All after time NMS 4 s.

    python3 work/I3_score/run2.py [--bags ...] [--variants XT0,XH,XHT]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
from common import BAGS, target_sources  # noqa: E402
from run import CFG_S, nms, write  # noqa: E402
from uwbfall import hitmodel as HM  # noqa: E402
from uwbfall.scoring import gt_records  # noqa: E402
from uwbfall.sources import events_for  # noqa: E402

FOLDS = ["0", "1", "2", "3"]


FEAT = "feat"


def load_target(bag, t, gts, durs):
    F = pickle.load(open(HERE / "cache" / f"{FEAT}_{bag}_{t}.pkl", "rb"))
    r = F["rows_te"]
    Y = HM.hit_labels(r.stream_id.values, r.t_start.values, r.t_end.values, gts[t])
    return {"X": F["Xte"], "rows": r.drop(columns="row"), "Y": Y}


def fit_choose(parts, gt, dur, cfg_s, use_hit=True, groups=None, tau_parts=None):
    return HM.fit_nested(parts, gt, dur, cfg_s, nms, groups=groups, tau_parts=tau_parts, use_hit=use_hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", default=",".join(BAGS))
    ap.add_argument("--variants", default="XT0,XH,XHT")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pool-bags", action="store_true", help="train the hit model on the rows of all bags")
    ap.add_argument("--fixed-tau", type=float, default=None,
                    help="XHT uses this fixed tau (F1*/2 rule) instead of the nested choice")
    ap.add_argument("--feat", default="feat", help="feature cache prefix: feat (reference candidates, run.py) | "
                    "featR (recall-oriented candidates, run3.py)")
    a = ap.parse_args()
    global FEAT
    FEAT = a.feat
    V = a.variants.split(",")
    gts, durs = {}, {}
    for t in FOLDS + ["official"]:
        _tr, _e, te, evte = target_sources(t)
        gts[t] = gt_records(events_for(te, evte), ignore_flagged=True)
        durs[t] = {s.sid: s.duration for s in te}
    info = {}
    bags = a.bags.split(",")
    TT = {bb: {t: load_target(bb, t, gts, durs) for t in FOLDS + ["official"]} for bb in BAGS}
    for bag in bags:
        T = TT[bag]
        for t in FOLDS + ["official"]:
            train = [f for f in FOLDS if f != t]
            pb = BAGS if a.pool_bags else [bag]
            parts = [TT[bb][f] for f in train for bb in pb]
            groups = [f for f in train for bb in pb]
            tau_parts = [i for i, (f, bb) in enumerate((f, bb) for f in train for bb in pb) if bb == bag]
            gt = sum((gts[f] for f in train), [])
            dur = {k: v for f in train for k, v in durs[f].items()}
            tgt = T[t]
            ref = nms(tgt["rows"])
            rec = {"train_folds": train}
            if "XT0" in V:
                _, tau0, c0, n = fit_choose([T[f] for f in train], gt, dur, CFG_S, use_hit=False)
                write(bag, "XT0" + a.tag, t, ref[ref.score >= tau0])
                rec.update(tau_XT0=tau0, curve_XT0=c0, rows=n)
            if "XH" in V or "XHT" in V:
                model, tau, c, n = fit_choose(parts, gt, dur, CFG_S, use_hit=True, groups=groups, tau_parts=tau_parts)
                if a.fixed_tau is not None:
                    tau = a.fixed_tau
                h = model.predict(tgt["X"])
                H = nms(tgt["rows"].assign(score=np.round(h, 6)))
                if "XH" in V:
                    write(bag, "XH" + a.tag, t, H)
                if "XHT" in V:
                    write(bag, "XHT" + a.tag, t, H[H.score >= tau])
                rec.update(tau_XHT=tau, curve_XHT=c, rows=n, importance=model.importance())
                from sklearn.metrics import roc_auc_score
                Y = tgt["Y"]; mk = Y.train_ok.values
                rec["target_auc"] = {str(th): (float(roc_auc_score(Y[f"hit{th}"][mk], h[mk])),
                                               float(roc_auc_score(Y[f"hit{th}"][mk], tgt["rows"].score.values[mk])))
                                     for th in HM.THETAS if Y[f"hit{th}"][mk].nunique() > 1}
            info[f"{bag}|{t}"] = rec
            print(bag, t, {k: v for k, v in rec.items() if k.startswith("tau") or k in ("rows", "target_auc")}, flush=True)
    (HERE / "out" / f"info_X{a.tag}.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

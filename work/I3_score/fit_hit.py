"""Fit the deployable hit model from ALL SR-CV OOF candidate tables (4 folds, pooled bags) and save the
state that `python -m uwbfall hitattach` / pipeline.attach_hit expects. Identical to the model run2.py uses for
the official val target with --pool-bags (same rows, same seeds).

    python3 work/I3_score/fit_hit.py <out.pkl> [--fixed-tau 0.46 | --nested] [--feat feat]
"""
import argparse
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
import run2  # noqa: E402
from common import BAGS, target_sources  # noqa: E402
from run import CFG_S, nms  # noqa: E402
from uwbfall import hitmodel as HM  # noqa: E402
from uwbfall.scoring import gt_records  # noqa: E402
from uwbfall.sources import events_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fixed-tau", type=float, default=None)
    ap.add_argument("--feat", default="feat")
    a = ap.parse_args()
    run2.FEAT = a.feat
    gts, durs = {}, {}
    for t in run2.FOLDS:
        _tr, _e, te, evte = target_sources(t)
        gts[t] = gt_records(events_for(te, evte), ignore_flagged=True)
        durs[t] = {s.sid: s.duration for s in te}
    parts = [run2.load_target(bb, f, gts, durs) for f in run2.FOLDS for bb in BAGS]
    groups = [f for f in run2.FOLDS for bb in BAGS]
    gt = sum(gts.values(), [])
    dur = {k: v for d in durs.values() for k, v in d.items()}
    model, tau, curve, n = HM.fit_nested(parts, gt, dur, CFG_S, nms, groups=groups)
    if a.fixed_tau is not None:
        tau = a.fixed_tau
    state = {"model": model, "tau": tau,
             "meta": {"trained_on": "SR-CV OOF test-part candidates, folds 0-3, bags " + ",".join(BAGS),
                      "rows": n, "tau_rule": "fixed F1*/2" if a.fixed_tau is not None else "nested",
                      "importance": model.importance()}}
    pickle.dump(state, open(a.out, "wb"))
    print(a.out, "tau", tau, n)


if __name__ == "__main__":
    main()

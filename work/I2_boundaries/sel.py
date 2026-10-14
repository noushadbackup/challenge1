"""(a) Boundary hypothesis selector (uwbfall.boundary.hypotheses / fit_selector / select) on the reference G7
candidates, per (bag, target). Trained on the bag's TRAIN-OOF candidates of the target (replay internals from
dump.py; targets = tIoU with unflagged training GT). The switch margin is chosen on the training rows only, by a
stream-grouped 4-fold inner CV (mean hit rate of the chosen hypothesis over theta .1/.3/.5, first max over
margins). Test: select -> to_preds -> time NMS 4 s (reference) -> preds/<bag>/S/<split>.csv

    python3 work/I2_boundaries/sel.py <targets> [bags] [--name S] [--margins 0,0.02,0.05,0.1]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BAGS, COLS, HERE, REF_CFG, oname, target_sources  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall.boundary import SEL_FEATURES, fit_selector, hypotheses, selector_targets, to_preds  # noqa: E402

B = REF_CFG["boundary"]


def unflagged(ev):
    fl = (ev.quality.astype(str) == "low") | (ev.needs_review.astype(str).str.lower() == "true")
    if "ignore" in ev:
        fl |= ev.ignore.astype(str).str.lower() == "true"
    return ev[~fl]


def chosen_iou(H, y, pred, margin):
    """tIoU of the hypothesis chosen per candidate (candidates with a GT only)."""
    H = H.assign(y=y, pred=pred)
    out = []
    for _i, g in H.groupby("row"):
        if not np.isfinite(g.y).any():
            continue
        ref = g[g.is_med == 1] if g.use.iloc[0] else g[g.is_det == 1]
        j = g.pred.idxmax()
        out.append(g.y[j] if g.pred[j] > ref.pred.iloc[0] + margin else ref.y.iloc[0])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("bags", nargs="?", default=",".join(BAGS))
    ap.add_argument("--name", default="S")
    ap.add_argument("--margins", default="0,0.02,0.05,0.1,inf")
    a = ap.parse_args()
    margins = [float(x) for x in a.margins.split(",")]
    log = {}
    for t in a.targets.split(","):
        tr, evtr, te, _ = target_sources(t)
        gt_tr = unflagged(evtr)
        dur = {s.sid: s.duration for s in te}
        Rtr = pickle.load(open(HERE / "cache" / f"rep_{t}_train.pkl", "rb"))
        Rte = pickle.load(open(HERE / "cache" / f"rep_{t}_test.pkl", "rb"))
        for bag in a.bags.split(","):
            Mtr = Rtr["cand"][bag].merge(Rtr["D"], on=["stream_id", "t_start", "t_end"], how="left")
            Htr = hypotheses(Mtr, B)
            ytr = selector_targets(Htr, Mtr, gt_tr)
            grp = Mtr.stream_id.values[Htr.row.values]
            oof = np.full(len(Htr), np.nan)
            for i_a, i_b in GroupKFold(4).split(Htr, groups=grp):
                m = fit_selector(Htr.iloc[i_a].reset_index(drop=True), ytr[i_a])
                oof[i_b] = m.predict(Htr.iloc[i_b][SEL_FEATURES])
            ref = chosen_iou(Htr, ytr, oof, np.inf)
            sc = {}
            for mg in margins:
                c = chosen_iou(Htr, ytr, oof, mg)
                sc[mg] = float(np.mean([(c >= th).mean() for th in (0.1, 0.3, 0.5)]))
            best = max(margins, key=lambda x: (sc[x], -margins.index(x)))
            best = [x for x in margins if sc[x] == max(sc.values())][0]
            model = fit_selector(Htr, ytr)
            Mte = Rte["cand"][bag].merge(Rte["D"], on=["stream_id", "t_start", "t_end"], how="left")
            from uwbfall.boundary import select
            s, e, _u = select(Mte, B, model, best)
            pr = CL.nms_time(to_preds(Mte, s, e, dur), REF_CFG["decode"]["nms_gap_s"]).reset_index(drop=True)[COLS]
            d = HERE / "preds" / bag / a.name
            d.mkdir(parents=True, exist_ok=True)
            pr.to_csv(d / f"{oname(t)}.csv", index=False)
            log[f"{bag}|{t}"] = {"margin": best, "inner_cv": sc,
                                 "ref_inner": float(np.mean([(ref >= th).mean() for th in (0.1, 0.3, 0.5)])),
                                 "n_train_cands": int(Htr.row.nunique()), "n_with_gt": int(len(ref))}
            print(f"{a.name} {bag} {t}: margin {best} inner-CV hit {sc[best]:.4f} vs ref {log[f'{bag}|{t}']['ref_inner']:.4f} "
                  f"({len(ref)} train cands with GT)", flush=True)
    f = HERE / "out" / f"sel_{a.name}.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(log)
    f.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()

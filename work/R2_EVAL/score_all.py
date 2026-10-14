"""Score candidate systems on SR-CV (pooled + per fold) and on official val; run promotion-rule comparisons.

    python3 work/R2_EVAL/score_all.py table  [bags]                 # -> out/table.json
    python3 work/R2_EVAL/score_all.py promote A B [bags]            # e.g. promote C2 C3 -> out/promote_C2_vs_C3.json
bags: comma list of bag names under preds/ (default: every complete bag). A list of bags = seeds; promote() then
averages counts over seeds (PLAN v2 §2: seed-average stochastic models before comparing).
Val is scored with the official CLI (python -m abc2027.evaluate ... --ignore-flagged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from srcv import scoring as S
from uwbfall.scoring import official_cli

HERE = Path(__file__).resolve().parent
PRED = HERE / "preds"
OUT = HERE / "out"


def complete(bag, c):
    d = PRED / bag / c
    return all((d / f"fold_{k}.csv").exists() for k in range(4)) and (d / "val.csv").exists()


def bags_for(c, bags=None):
    bags = bags or sorted(p.name for p in PRED.iterdir() if p.is_dir())
    return [b for b in bags if complete(b, c)]


def preds(bag, c):
    return {k: PRED / bag / c / f"fold_{k}.csv" for k in range(4)}


def val_cli(bag, c):
    f = PRED / bag / c / "val.csv"
    j = f.with_suffix(".cli.json")
    rep = official_cli(f, "val", j)
    return S.summary(rep)


def system_row(c, bags):
    per_bag = []
    for b in bags:
        P = preds(b, c)
        pooled = S.summary(S.score_pooled(P))
        folds = [S.summary(S.score(P[k], k))["rank_f1"] for k in range(4)]
        per_bag.append({"bag": b, "pooled": pooled, "folds": folds, "val": val_cli(b, c),
                        "noP3_3": S.rank_f1_by_subject_mask(P, ["Test_P3_3"])})
    ci = S.bootstrap_ci([preds(b, c) for b in bags])
    mean = lambda f: float(np.mean([f(r) for r in per_bag]))  # noqa: E731
    keys = list(per_bag[0]["pooled"])
    return {"system": c, "bags": bags, "per_bag": per_bag,
            "seedmean": {"pooled": {k: mean(lambda r: r["pooled"][k]) for k in keys},
                         "folds": [mean(lambda r: r["folds"][k]) for k in range(4)],
                         "val": {k: mean(lambda r: r["val"][k]) for k in keys},
                         "noP3_3": mean(lambda r: r["noP3_3"]),
                         "pooled_rank_sd_over_bags": float(np.std([r["pooled"]["rank_f1"] for r in per_bag], ddof=1))
                         if len(per_bag) > 1 else None},
            "bootstrap_seedavg_counts": ci}


def main():
    a = sys.argv[1:]
    OUT.mkdir(exist_ok=True)
    if a[0] == "table":
        systems = a[1].split(",") if len(a) > 1 else ["C0", "C1", "C2", "C3", "C4"]
        bags = a[2].split(",") if len(a) > 2 else None
        res = {}
        f = OUT / "table.json"
        if f.exists():
            res = json.loads(f.read_text())
        for c in systems:
            res[c] = r = system_row(c, bags_for(c, bags))
            m = r["seedmean"]
            print(f"{c}: bags {r['bags']} pooled {m['pooled']['rank_f1']:.4f} (sd {m['pooled_rank_sd_over_bags']}) "
                  f"folds {[round(x, 4) for x in m['folds']]} val {m['val']['rank_f1']:.4f} "
                  f"tp/fp/fn {m['pooled']['tp@0.3']:.1f}/{m['pooled']['fp@0.3']:.1f}/{m['pooled']['fn@0.3']:.1f}",
                  flush=True)
            f.write_text(json.dumps(res, indent=1, default=float))
    elif a[0] == "promote":
        A, B = a[1], a[2]
        bags = a[3].split(",") if len(a) > 3 else None
        ba = bags_for(A, bags)
        bb = bags_for(B, bags)
        assert ba == bb, (ba, bb)
        va = float(np.mean([val_cli(b, A)["rank_f1"] for b in ba]))
        vb = float(np.mean([val_cli(b, B)["rank_f1"] for b in bb]))
        arm = lambda c, bs: [preds(b, c) for b in bs] if len(bs) > 1 else preds(bs[0], c)  # noqa: E731
        rep = S.promote(arm(A, ba), arm(B, bb), n_boot=5000, val_A=va, val_B=vb)
        rep["A"], rep["B"], rep["bags"] = A, B, ba
        # diagnostic: same comparison with the Test_P3_3 cluster removed
        rep["noP3_3"] = {"A": float(np.mean([S.rank_f1_by_subject_mask(preds(b, A), ["Test_P3_3"]) for b in ba])),
                         "B": float(np.mean([S.rank_f1_by_subject_mask(preds(b, B), ["Test_P3_3"]) for b in bb]))}
        (OUT / f"promote_{A}_vs_{B}.json").write_text(json.dumps(rep, indent=1, default=float))
        print(f"{A} -> {B}: delta {rep['delta']:+.4f} CI {np.round(rep['delta_ci95'], 4).tolist()} "
              f"P {rep['p_delta_gt_0']:.3f} folds {rep['n_folds_improved']}/4 "
              f"{[round(r['delta'], 4) for r in rep['per_fold']]} val {va:.4f}->{vb:.4f} "
              f"noP3_3 {rep['noP3_3']['A']:.4f}->{rep['noP3_3']['B']:.4f} PROMOTE={rep['promote']}", flush=True)


if __name__ == "__main__":
    main()

"""Tune each recovery rule on every bundle's own training OOF, apply the chosen config to that bundle's held-out
streams (fold test streams, or val for official_train), score, and test against the base under the promotion rule.

Variants:  sec | amax | raw  (rule; boundary mode det/rep/seg is part of the train-only grid)
           sec_rep | amax_rep | amax_seg  (boundary mode fixed, rest of the grid tuned on train)
Selection (pre-registered, rules.select): max training-OOF delta, ties -> fewest added rows; no gain -> base.

    python3 work/R3_recover/run.py [variant ...]
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

import srcv

import rules as RL

HERE = Path(__file__).resolve().parent
BASE = Path("/home/user/challenge1/work/STRICT/results/norep")
TARGETS = ["fold_0", "fold_1", "fold_2", "fold_3", "official_train"]
VARIANTS = {
    "sec": ("sec", ("det", "rep", "seg")), "amax": ("amax", ("det", "rep", "seg")),
    "raw": ("raw", ("det", "rep", "seg")),
    "sec_rep": ("sec", ("rep",)), "amax_rep": ("amax", ("rep",)), "amax_seg": ("amax", ("seg",)),
    "segcrop": ("amax", ("seg",), "segcrop"),
}


def bounds(target, part):
    with open(RL.R / target / f"bounds_{part}.pkl", "rb") as f:
        return pickle.load(f)


def run(name: str) -> dict:
    rule, modes, *grid = VARIANTS[name]
    grid = grid[0] if grid else None
    res = HERE / "results" / name
    res.mkdir(parents=True, exist_ok=True)
    chosen, preds, trainrep = {}, {}, {}
    for t in TARGETS:
        dtr = RL.load(t, "train")
        tab, base = RL.tune(dtr, rule, modes=modes, bounds=bounds(t, "train"), grid=grid)
        tab.to_csv(RL.R / t / f"tab_{name}.csv", index=False)
        sel = RL.select(tab)
        chosen[t] = sel
        best = tab.delta.max()
        trainrep[t] = {"base_rank": RL.rank(base), "best_delta": float(best), "chosen": sel}
        dte = RL.load(t, "test")
        if sel is None:
            rows = dte["rows"].copy()
            n_add = 0
        else:
            prm = {k: v for k, v in sel.items() if k != "mode"}
            ex = RL.propose(dte, rule, prm, dte["postproc"])
            rows = RL.add_rows(dte, ex, mode=sel["mode"], bounds=bounds(t, "test"))
            n_add = len(rows) - len(dte["rows"])
        trainrep[t]["n_add_test"] = n_add
        out = res / ("val.csv" if t == "official_train" else f"fold_{t[-1]}.csv")
        rows[RL.COLS].to_csv(out, index=False)
        if t != "official_train":
            preds[int(t[-1])] = out
        print(f"[{name}] {t}: train base {RL.rank(base):.4f} best delta {best:+.4f} chosen {sel} -> {n_add} rows added",
              flush=True)
    A = {k: BASE / f"fold_{k}.csv" for k in range(4)}
    pr = srcv.promote(A, preds, val_A=BASE / "val.csv", val_B=res / "val.csv")
    pooled = srcv.score_pooled(preds)
    s = {"variant": name, "rule": rule, "modes": modes, "train": trainrep, "pooled_rank_f1": pooled["ranking_score"]["f1_mean"],
         "tp_fp_fn@0.3": [pooled["by_theta"]["0.3"][k] for k in ("tp", "fp", "fn")], "promotion_vs_norep": pr}
    (res / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(f"[{name}] pooled {s['pooled_rank_f1']:.4f} {s['tp_fp_fn@0.3']} delta {pr['delta']:+.4f} "
          f"CI {pr['delta_ci95']} P {pr['p_delta_gt_0']:.3f} folds {pr['n_folds_improved']} "
          f"val {pr['val_A']:.4f}->{pr['val_B']:.4f} promote {pr['promote']}", flush=True)
    return s


if __name__ == "__main__":
    for v in sys.argv[1:] or list(VARIANTS):
        run(v)

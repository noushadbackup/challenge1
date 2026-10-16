"""Tune the verifier cut on TRAINING-fold OOF only, apply it to the base predictions, score.

usage: python apply.py <tag> <variant> [--seeds 0,1] [--sets fold0,...,official] [--rule cut|cut2]
  rule cut : drop a base row if p_verifier < tau; tau = first maximiser of the official ranking F1 of the bundle's
             OOF final rows (training streams; scored with the training events, --ignore-flagged) over
             tau in {0, 0.025, ..., 0.7} (tau=0 = keep everything = base).
  rule cut2: drop if p < tau AND detector score < sigma; (tau, sigma) on the same training OOF grid
             (sigma in {0.6, 0.7, 0.8, 0.9, 1.01}).
p_verifier on the training rows = subject-grouped inner OOF of cnn.py (mode inner), on test rows = the model trained
on all training candidates (mode full); seeds are averaged.
Writes results/<variant>/{fold_k,val}.csv (+ tune.json); the test streams / val labels are only read for scoring.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import srcv
from srcv import scoring as SC
from abc2027 import evaluate as E
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
BASE = "/home/user/challenge1/work/STRICT/results/norep"
COLS = ["stream_id", "t_start", "t_end", "score", "t_alarm"]


def train_rank_f1(df, S):
    if S.startswith("fold"):
        return srcv.score(df[COLS], int(S[4:]), "train")["ranking_score"]["f1_mean"]
    dur = E.stream_durations(PKG / "streams" / "train")
    gt = [g for g in E.load_ground_truth(PKG / "streams_events.csv", ignore_flagged=True) if g["stream_id"] in dur]
    return E.evaluate(gt, SC._records(df[COLS], dur), dur, E.PRIMARY_THETA)["ranking_score"]["f1_mean"]


def load_p(tag, S, mode, seeds):
    ps = [pd.read_csv(f"{HERE}/probs/{tag}/{S}_{mode}_s{s}.csv") for s in seeds]
    df = ps[0].copy()
    df["p"] = np.mean([d.p.values for d in ps], 0)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("variant")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--sets", default="fold0,fold1,fold2,fold3,official")
    ap.add_argument("--rule", default="cut")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = f"{HERE}/results/{a.variant}"
    os.makedirs(out_dir, exist_ok=True)
    taus = np.round(np.arange(0, 0.7001, 0.025), 3)
    sigmas = [0.6, 0.7, 0.8, 0.9, 1.01] if a.rule == "cut2" else [1.01]
    info = {}
    for S in a.sets.split(","):
        tr = load_p(a.tag, S, "inner", seeds)
        fin = tr[tr.src == "final"].reset_index(drop=True)
        auc_tr = roc_auc_score(tr.hit, tr.p)
        auc_fin = roc_auc_score(fin.hit, fin.p) if 0 < fin.hit.sum() < len(fin) else float("nan")
        best, table = None, []
        base_f1 = train_rank_f1(fin, S)
        for sg in sigmas:
            for t in taus:
                keep = ~((fin.p < t) & (fin.score < sg))
                f1 = train_rank_f1(fin[keep], S) if (~keep).any() else base_f1
                table.append((float(t), sg, f1, int((~keep).sum())))
                if best is None or f1 > best[2] + 1e-12:
                    best = (float(t), sg, f1, int((~keep).sum()))
        t, sg = best[0], best[1]
        te = load_p(a.tag, S, "full", seeds)
        drop = (te.p < t) & (te.score < sg)
        name = f"fold_{S[4:]}.csv" if S.startswith("fold") else "val.csv"
        te.loc[~drop, COLS].to_csv(f"{out_dir}/{name}", index=False)
        info[S] = {"tau": t, "sigma": sg, "train_oof_rank_f1_base": base_f1, "train_oof_rank_f1_tuned": best[2],
                   "train_oof_dropped": best[3], "train_auc_all": auc_tr, "train_auc_final_rows": auc_fin,
                   "test_dropped": int(drop.sum()), "test_dropped_hits": int((drop & (te.hit == 1)).sum()),
                   "test_auc(analysis only)": roc_auc_score(te.hit, te.p) if 0 < te.hit.sum() < len(te) else None}
        print(S, json.dumps(info[S]), flush=True)
    json.dump(info, open(f"{out_dir}/tune.json", "w"), indent=1)


if __name__ == "__main__":
    main()

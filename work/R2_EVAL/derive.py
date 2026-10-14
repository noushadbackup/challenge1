"""Derive candidate systems C0..C4 (and boundary variants) from fitted uwbfall bundles, without refitting.

All candidates of one bag share the same fitted models (same seeds), so they differ ONLY in the stage under test:
  C0  HGB alone, detector bounds            (operating point tuned on the HGB train OOF)
  C1  TCN bag alone, detector bounds        (operating point tuned on the TCN train OOF)
  C2  0.75 TCN + 0.25 HGB, detector bounds  (operating point tuned on the blended train OOF)
  C3  C2 + replay B                         (= uwbfall default)
  C4  C1 + replay B
HGB/PCA come from the uwb_default bundle of the same target (deterministic, identical in every bag); the TCN bag
(3 members + its train OOF) comes from <bag>. bag "uwb_default" is the original 3-member bag (E4 seeds).

    python3 work/R2_EVAL/derive.py <bag> [targets 0,1,2,3,official]
Writes work/R2_EVAL/preds/<bag>/<C>/{fold_k,val}.csv, cache/probs_<bag>_<target>.npz, cache/D_<bag>_<target>_<C>.csv
(raw replay table per candidate, for the mechanism analysis) and cache/ops_<bag>_<target>.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import uwbfall
from uwbfall.boundary import refine, replay_stream, to_preds
from uwbfall.config import load_config
from uwbfall.decode import events_df, to_events, tune
from uwbfall.extract import get_features
from uwbfall.features import design, stream_blocks
from uwbfall.models import build_tcn, predict_seq, setup_torch, standardise
from uwbfall.scoring import gt_records
from uwbfall.sources import events_for
from srcv import bridge, core as C

HERE = Path(__file__).resolve().parent
RUNS = C.DATA / "runs"
SYSTEMS = {  # name: (prob source, replay)
    "C0": ("hgb", False), "C1": ("tcn", False), "C2": ("blend", False), "C3": ("blend", True), "C4": ("tcn", True),
}


def target_sources(t):
    if t == "official":
        return (uwbfall.split_sources("train"), C.real_events("train"), uwbfall.split_sources("val"), "val")
    k = int(t)
    return (bridge.pipeline_sources(k, "train"), C.events(k, "train"), bridge.pipeline_sources(k, "test"),
            f"fold_{k}")


def load_models(bag, tdir):
    base = RUNS / "uwb_default" / tdir
    bagd = RUNS / bag / tdir
    cfg = json.loads((base / "config.json").read_text())
    with open(base / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(base / "hgb.pkl", "rb") as f:
        hgb = pickle.load(f)
    z = np.load(bagd / "tcn_norm.npz")
    mu, sd = z["mu"], z["sd"]
    bcfg = json.loads((bagd / "config.json").read_text())
    tcns = []
    for s in range(bcfg["tcn"]["seeds"]):
        m = build_tcn(len(mu), cfg)
        m.load_state_dict(torch.load(bagd / f"tcn_{s}.pt", weights_only=True))
        m.eval()
        tcns.append(m)
    o0, o1 = np.load(base / "oof.npz"), np.load(bagd / "oof.npz")
    assert (o0["sid"] == o1["sid"]).all() and (o0["y"] == o1["y"]).all()
    return cfg, pca, hgb, mu, sd, tcns, o0["sid"], o0["oof_hgb"], o1["oof_tcn"]


def blend(w, pt, ph):
    return w * pt + (1 - w) * ph


def main():
    bag = sys.argv[1]
    targets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2", "3", "official"]
    (HERE / "cache").mkdir(exist_ok=True)
    for t in targets:
        t0 = time.time()
        tdir = "official_train" if t == "official" else f"fold_{t}"
        tr, ev, te, oname = target_sources(t)
        cfg, pca, hgb, mu, sd, tcns, sid, oof_h, oof_t = load_models(bag, tdir)
        cfg = load_config({k: v for k, v in cfg.items()})  # validate
        _lim = setup_torch(cfg)  # noqa: F841
        w = cfg["blend"]["w_tcn"]
        # operating points on train OOF (official evaluator, same grid / first-max rule as uwbfall.fit)
        evs = events_for(sorted(tr, key=lambda s: s.sid), ev)
        gt = gt_records(evs, ignore_flagged=True)
        dur_tr = {s.sid: s.duration for s in tr}
        oofs = {"hgb": oof_h, "tcn": oof_t, "blend": blend(w, oof_t, oof_h)}
        ops = {}
        for src, p in oofs.items():
            probs_by = {s: p[sid == s] for s in pd.unique(sid)}
            sc, g, _ = tune(probs_by, gt, dur_tr, cfg)
            ops[src] = {"op": list(g), "oof_rank_f1": sc}
        (HERE / "cache" / f"ops_{bag}_{t}.json").write_text(json.dumps(ops, indent=1))
        # test / val streams
        pf = HERE / "cache" / f"probs_{bag}_{t}.npz"
        P = dict(np.load(pf)) if pf.exists() else {}
        durations = {s.sid: s.duration for s in te}
        rows = {c: [] for c in SYSTEMS}
        Ds = {c: [] for c in SYSTEMS}
        for s in sorted(te, key=lambda s: s.sid):
            F = get_features(s, cfg)
            if f"{s.sid}|hgb" not in P:
                blk = stream_blocks(F, cfg)
                xp, xb, _ = design(blk, pca, cfg)
                P[f"{s.sid}|hgb"] = hgb.predict_proba(xp)[:, 1]
                Xs = standardise(np.hstack([xp.astype(np.float32), xb]), mu, sd, cfg["tcn"]["clip_z"])
                P[f"{s.sid}|tcn"] = np.mean([predict_seq(m, Xs) for m in tcns], 0)
            ph, pt = P[f"{s.sid}|hgb"], P[f"{s.sid}|tcn"]
            pr = {"hgb": ph, "tcn": pt, "blend": blend(w, pt, ph)}
            replayed = {}
            for c, (src, rep) in SYSTEMS.items():
                cand = events_df(to_events(s.sid, pr[src], *ops[src]["op"]))
                if not len(cand):
                    continue
                if rep:
                    if src not in replayed:
                        replayed[src] = replay_stream(s, F["prof_02"], cand, cfg)
                    D = replayed[src]
                    a, e, use = refine(D, cfg["boundary"])
                    rows[c].append(to_preds(D, a, e, durations))
                    Ds[c].append(D.assign(use=use, r_s=a, r_e=e))
                else:
                    rows[c].append(to_preds(cand, cand.t_start.values, cand.t_end.values, durations))
            s.release()
        np.savez_compressed(pf, **P)
        cols = ["stream_id", "t_start", "t_end", "score", "t_alarm"]
        for c in SYSTEMS:
            d = HERE / "preds" / bag / c
            d.mkdir(parents=True, exist_ok=True)
            df = pd.concat(rows[c], ignore_index=True)[cols] if rows[c] else pd.DataFrame(columns=cols)
            df.to_csv(d / f"{oname}.csv", index=False)
            if Ds[c]:
                pd.concat(Ds[c], ignore_index=True).to_csv(HERE / "cache" / f"D_{bag}_{t}_{c}.csv", index=False)
        print(f"[{bag}] target {t}: ops {json.dumps({k: v['op'] for k, v in ops.items()})} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

"""I1 stage 2: decode variants from fitted bundles (no TCN refit), per (bag, target).

  REF  0.75 TCN + 0.25 HGB, op tuned on blended train OOF, replay gate 0.3       (= R2_EVAL C3G3; byte-checked)
  G    REF probabilities x P(fall clip) of the block's segment (segment classifier, subject-grouped OOF on train)
  D    0.75 TCN + 0.25 P(fall) of the 3-class HGB (fall / post-fall-in-clip / other)
  DG   D x P(fall clip)
Each variant re-tunes its operating point on ITS OWN train OOF (official evaluator, uwbfall grid, first max), then
runs the annotator replay (reference boundary stage, gate 0.3) on its candidates. NMS variants are applied
post hoc by apply_nms.py.

    python3 work/I1_clip/variants.py <bag> <targets comma> [variants comma] [--gate-set NAME]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import uwbfall
from uwbfall import clip as CL
from uwbfall.boundary import refine, replay_stream, to_preds
from uwbfall.config import load_config
from uwbfall.decode import events_df, to_events, tune
from uwbfall.extract import get_features
from uwbfall.models import setup_torch
from uwbfall.scoring import gt_records
from uwbfall.sources import events_for
from srcv import bridge, core as C

HERE = Path(__file__).resolve().parent
R2 = HERE.parent / "R2_EVAL"
RUNS = C.DATA / "runs"
REFB = load_config(None)["boundary"]  # reference boundary stage (gate 0.3)


def target_sources(t):
    if t == "official":
        return uwbfall.split_sources("train"), C.real_events("train"), uwbfall.split_sources("val"), "val"
    k = int(t)
    return bridge.pipeline_sources(k, "train"), C.events(k, "train"), bridge.pipeline_sources(k, "test"), f"fold_{k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("targets")
    ap.add_argument("variants", nargs="?", default="REF,G,D,DG")
    ap.add_argument("--gate-set", default="gate")
    ap.add_argument("--gate-mode", default="mult")
    ap.add_argument("--gate-floor", type=float, default=0.0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    variants = a.variants.split(",")
    for t in a.targets.split(","):
        t0 = time.time()
        tdir = "official_train" if t == "official" else f"fold_{t}"
        base, bagd = RUNS / "uwb_default" / tdir, RUNS / a.bag / tdir
        cfg = load_config(json.loads((base / "config.json").read_text()))
        cfg["boundary"] = dict(REFB)
        _lim = setup_torch(cfg)  # noqa: F841
        w = cfg["blend"]["w_tcn"]
        tr, ev, te, oname = target_sources(t)
        tr = sorted(tr, key=lambda s: s.sid)
        te = sorted(te, key=lambda s: s.sid)
        o0, o1 = np.load(base / "oof.npz"), np.load(bagd / "oof.npz")
        sid = o0["sid"]
        assert (sid == o1["sid"]).all()
        Pt = dict(np.load(R2 / "cache" / f"probs_{a.bag}_{t}.npz"))
        prep = pickle.load(open(HERE / "cache" / f"prep_{t}.pkl", "rb"))
        h3 = np.load(HERE / "cache" / f"hgb3_{t}.npz")
        assert (h3["sid"] == sid).all()
        gates = prep.get(a.gate_set)  # {sid: per-block P(fall clip)} for train (OOF) and test streams
        evs = events_for(tr, ev)
        gt = gt_records(evs, ignore_flagged=True)
        dur_tr = {s.sid: s.duration for s in tr}
        durations = {s.sid: s.duration for s in te}

        def probs(v, oof: bool, s=None):
            if oof:
                ph, pt, p3 = o0["oof_hgb"], o1["oof_tcn"], h3["oof3"][:, 1]
            else:
                ph, pt, p3 = Pt[f"{s}|hgb"], Pt[f"{s}|tcn"], h3[f"{s}|hgb3"][:, 1]
            p = w * pt + (1 - w) * (p3 if v.startswith("D") else ph)
            if "G" in v:
                if oof:
                    g = np.concatenate([gates[x] for x in pd.unique(sid)])
                else:
                    g = gates[s]
                p = CL.gate_probs(p, g, a.gate_mode, a.gate_floor)
            return p

        ops = {}
        for v in variants:
            p = probs(v, True)
            pb = {s: p[sid == s] for s in pd.unique(sid)}
            sc, g, _ = tune(pb, gt, dur_tr, cfg)
            ops[v] = {"op": list(g), "oof_rank_f1": sc}
        rows = {v: [] for v in variants}
        cache_rep = {}
        for s in te:
            F = get_features(s, cfg)
            for v in variants:
                cand = events_df(to_events(s.sid, probs(v, False, s.sid), *ops[v]["op"]))
                if not len(cand):
                    continue
                key = cand.to_csv(index=False)
                if key not in cache_rep:
                    cache_rep[key] = replay_stream(s, F["prof_02"], cand, cfg)
                D = cache_rep[key]
                st, en, _use = refine(D, cfg["boundary"])
                rows[v].append(to_preds(D, st, en, durations))
            s.release()
            cache_rep.clear()
        cols = ["stream_id", "t_start", "t_end", "score", "t_alarm"]
        for v in variants:
            d = HERE / "preds" / a.bag / (v + a.suffix if v != "REF" else "REFx")
            d.mkdir(parents=True, exist_ok=True)
            df = pd.concat(rows[v], ignore_index=True)[cols] if rows[v] else pd.DataFrame(columns=cols)
            df.to_csv(d / f"{oname}.csv", index=False)
        of = HERE / "cache" / f"ops_{a.bag}_{t}{a.suffix}.json"
        allops = json.loads(of.read_text()) if of.exists() else {}
        allops.update(ops)
        of.write_text(json.dumps(allops, indent=1))
        print(f"[{a.bag}] {t}: ops {json.dumps({k: v['op'] for k, v in ops.items()})} "
              f"oof {json.dumps({k: round(v['oof_rank_f1'], 4) for k, v in ops.items()})} ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()

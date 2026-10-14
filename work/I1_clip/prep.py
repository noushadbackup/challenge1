"""I1 stage 1, per target (SR-CV fold k or 'official'): everything that does not depend on the TCN bag.

  * junction statistic on every train/test stream, threshold for fall-junction recall on the TRAIN streams
  * segment tables + labels, LightGBM segment classifier: OOF P(fall clip) on train streams (bundle folds),
    final model -> P(fall clip) on test streams
  * (with --hgb3) 3-class HGB (fall / post-fall-in-clip / other) on the prototype 64 features: OOF on train
    (bundle folds) + final -> P(fall) on test streams

    python3 work/I1_clip/prep.py <target> [--hgb3] [--recall 0.95]
-> cache/prep_<target>.pkl  (and cache/hgb3_<target>.npz)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import uwbfall
from uwbfall import clip as CL
from uwbfall.config import load_config
from uwbfall.extract import get_features
from uwbfall.features import block_labels, design, proto_frame, stream_blocks
from uwbfall.models import setup_torch
from uwbfall.sources import events_for
from srcv import bridge, core as C

HERE = Path(__file__).resolve().parent
RUNS = C.DATA / "runs"


def target_sources(t):
    if t == "official":
        return uwbfall.split_sources("train"), C.real_events("train"), uwbfall.split_sources("val"), "val"
    k = int(t)
    return bridge.pipeline_sources(k, "train"), C.events(k, "train"), bridge.pipeline_sources(k, "test"), f"fold_{k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--hgb3", action="store_true")
    ap.add_argument("--recall", type=float, default=0.95)
    ap.add_argument("--min_sep", type=float, default=1.0)
    a = ap.parse_args()
    t = a.target
    t0 = time.time()
    tdir = "official_train" if t == "official" else f"fold_{t}"
    base = RUNS / "uwb_default" / tdir
    cfg = load_config(json.loads((base / "config.json").read_text()))
    _lim = setup_torch(cfg)  # noqa: F841
    with open(base / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    oof = np.load(base / "oof.npz")
    fold_of = dict(zip(oof["sid"], oof["fold"]))
    tr, ev, te, oname = target_sources(t)
    tr = sorted(tr, key=lambda s: s.sid)
    te = sorted(te, key=lambda s: s.sid)
    evs = events_for(tr, ev)
    # pass 1: statistic + light per-stream arrays
    S = {}
    Xp3, Y3, SID3 = [], [], []
    for part, srcs in (("train", tr), ("test", te)):
        for s in srcs:
            F = get_features(s, cfg)
            blk = stream_blocks(F, cfg)
            d = proto_frame(blk, pca)
            fp = d[[f"fp{i}" for i in range(cfg["features"]["pca_components"])]].values
            stat = CL.junction_stat(F["prof_02"])
            comp = CL.composition_of(s) if part == "train" or t != "official" else CL.composition_of(s)
            S[s.sid] = {"part": part, "stat": stat, "n": blk["n"], "comp": comp, "fp": fp, "base": blk["base"],
                        "F": None}
            if a.hgb3:
                xp, _xb, _ = design(blk, pca, cfg)
                if part == "train":
                    y, _ = block_labels(evs, s.sid, blk["n"])
                    pf = CL.postfall_labels(evs, comp, s.sid, blk["n"])
                    y3 = np.where(y, 1, np.where(pf, 2, 0)).astype(np.int8)
                    Xp3.append(xp); Y3.append(y3); SID3.append(np.full(blk["n"], s.sid))
                else:
                    S[s.sid]["xp"] = xp
            s.release()
    trs = [s.sid for s in tr]
    thr = CL.junction_threshold({k: S[k]["stat"] for k in trs}, {k: S[k]["comp"] for k in trs}, a.recall)
    diag = {p: CL.junction_recall({k: v["stat"] for k, v in S.items() if v["part"] == p},
                                  {k: v["comp"] for k, v in S.items() if v["part"] == p}, thr, a.min_sep)
            for p in ("train", "test")}
    print(f"[{t}] junction thr {thr:.3f} (train fall-junction recall target {a.recall}) diag {json.dumps(diag)}",
          flush=True)
    # pass 2: segment tables (needs F again for the static profile and repeats)
    tabs = []
    for s in tr + te:
        F = get_features(s, cfg)
        v = S[s.sid]
        pk = CL.junction_peaks(v["stat"], thr, a.min_sep)
        segs = CL.segments_from_peaks(v["n"], pk)
        tab = CL.segment_table(F, v["base"], v["fp"], v["stat"], segs)
        tab.insert(0, "sid", s.sid)
        tab["a"], tab["b"] = segs[:, 0], segs[:, 1]
        tab["y"] = CL.segment_labels(segs, v["comp"])
        tab["fold"] = fold_of.get(s.sid, -1) if v["part"] == "train" else -1
        tab["part"] = v["part"]
        v["peaks_t"] = pk * CL.STEP
        tabs.append(tab)
        del F
    T = pd.concat(tabs, ignore_index=True)
    feats = [c for c in T.columns if c not in ("sid", "a", "b", "y", "fold", "part")]
    # segment classifier: OOF on train (bundle folds) + final -> test
    T["p_adl"] = T["p_fall"] = T["p_other"] = np.nan
    trm = (T.part == "train").values
    P = np.zeros((len(T), 3))
    for k in sorted(T.fold[trm].unique()):
        m_tr = trm & (T.fold.values != k)
        m_ho = trm & (T.fold.values == k)
        clf = CL.lgbm_segclf(cfg).fit(T.loc[m_tr, feats], T.y[m_tr])
        P[m_ho] = clf.predict_proba(T.loc[m_ho, feats])
    clf = CL.lgbm_segclf(cfg).fit(T.loc[trm, feats], T.y[trm])
    P[~trm] = clf.predict_proba(T.loc[~trm, feats])
    T[["p_adl", "p_fall", "p_other"]] = P
    from sklearn.metrics import roc_auc_score
    rep = {}
    for p, m in (("train_oof", trm), ("test", ~trm)):
        yy = T.y.values[m] == 1
        rep[p] = {"auc_fall_vs_rest": float(roc_auc_score(yy, P[m, 1])) if yy.any() and (~yy).any() else None,
                  "n_seg": int(m.sum()), "n_fall_seg": int(yy.sum()),
                  "acc": float((P[m].argmax(1) == T.y.values[m]).mean())}
    print(f"[{t}] segclf {json.dumps(rep)}", flush=True)
    out = {"thr": thr, "diag": diag, "segclf": rep, "feats": feats, "table": T,
           "peaks": {k: v["peaks_t"] for k, v in S.items()}, "n": {k: v["n"] for k, v in S.items()},
           "train": trs, "test": [s.sid for s in te], "oname": oname}
    (HERE / "cache").mkdir(exist_ok=True)
    with open(HERE / "cache" / f"prep_{t}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"[{t}] prep done {time.time() - t0:.0f}s", flush=True)
    if a.hgb3:
        from uwbfall.models import fit_hgb
        X, y3, sid = np.vstack(Xp3), np.concatenate(Y3), np.concatenate(SID3)
        assert (sid == oof["sid"]).all()
        fold = oof["fold"]
        oof3 = np.zeros((len(y3), 3))
        for k in range(cfg["cv"]["n_folds"]):
            m = fit_hgb(X[fold != k], y3[fold != k], cfg)
            oof3[fold == k] = m.predict_proba(X[fold == k])
            print(f"[{t}] hgb3 fold {k} ({time.time() - t0:.0f}s)", flush=True)
        m = fit_hgb(X, y3, cfg)
        with open(HERE / "cache" / f"hgb3_{t}.pkl", "wb") as f:
            pickle.dump(m, f)
        test = {f"{sid}|hgb3": m.predict_proba(v["xp"]) for sid, v in S.items() if v["part"] == "test"}
        np.savez_compressed(HERE / "cache" / f"hgb3_{t}.npz", sid=sid, oof3=oof3, y3=y3, **test)
        print(f"[{t}] hgb3 done {time.time() - t0:.0f}s, post-fall blocks {int((y3 == 2).sum())}", flush=True)


if __name__ == "__main__":
    main()

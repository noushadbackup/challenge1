"""fit(train_sources, events_df, config) -> bundle directory;  predict(bundle, sources) -> DataFrame.

Training (all choices on the training sources only):
  1. features per stream (cached), train-only PCA fingerprint -> 64 prototype + 48 band channels
  2. GroupKFold(4) by stream -> OOF probabilities of the HGB and of the TCN bag (3 seeds, fixed epochs)
  3. blend OOF (w*TCN + (1-w)*HGB) -> operating point (thr, smooth, min_dur, gap) on the official evaluator
  4. final HGB + TCN bag on all training streams
Prediction, one stream at a time: features -> HGB + TCN bag -> blend -> candidate events ->
annotator replay on raw crops (config B) -> validated submission rows.
"""
from __future__ import annotations

import json
import pickle
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import GroupKFold

from . import __version__
from .boundary import refine, replay_stream, to_preds
from .config import check_implemented, config_hash, load_config
from .decode import decode_streams, events_df, to_events, tune
from .extract import EXTRACTOR_VERSION, get_features
from .features import block_labels, design, fit_pca, stream_blocks
from .models import build_tcn, fit_hgb, predict_seq, seqs, setup_torch, standardise, train_tcn
from .scoring import gt_records, score_inproc, summary
from .sources import StreamSource, events_for


def _logger(path: Path | None):
    t0 = time.time()
    fh = open(path, "a") if path else None

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB] {msg}"
        print(line, file=sys.stderr, flush=True)
        if fh:
            fh.write(line + "\n")
            fh.flush()
    return log


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def _blend(cfg: dict, p_tcn, p_hgb):
    w = cfg["blend"]["w_tcn"]
    if p_tcn is None:
        return p_hgb
    if p_hgb is None:
        return p_tcn
    return w * p_tcn + (1 - w) * p_hgb  # same expression/dtypes as E4 compare.py (float32 TCN, float64 HGB)


# ============================================================================ fit
def fit(train_sources: list[StreamSource], events: pd.DataFrame | None, config=None, out_dir=None, log=None) -> Path:
    cfg = load_config(config)
    check_implemented(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or _logger(out / "train.log")
    _lim = setup_torch(cfg)  # noqa: F841
    t_start = time.time()
    timings = {}
    sources = sorted(train_sources, key=lambda s: s.sid)
    ev = events_for(sources, events)
    log(f"fit: {len(sources)} training streams, {len(ev)} events, config {config_hash(cfg)}")

    # 1. features, one stream at a time
    blks, keys = [], {}
    for s in sources:
        F = get_features(s, cfg, log)
        blks.append(stream_blocks(F, cfg))
        keys[s.sid] = s.content_key()
        s.release()
        del F
    timings["features_s"] = round(time.time() - t_start, 1)
    pca = fit_pca([b["amp"] for b in blks], cfg)
    Xp, Xb, Y, SID, cols = [], [], [], [], None
    for s, b in zip(sources, blks):
        xp, xb, cols = design(b, pca, cfg)
        y, _fl = block_labels(ev, s.sid, b["n"])
        Xp.append(xp); Xb.append(xb); Y.append(y.astype(np.int8)); SID.append(np.full(b["n"], s.sid))
        b.pop("amp")
    Xp, Xb, y, sid = np.vstack(Xp), np.vstack(Xb), np.concatenate(Y), np.concatenate(SID)
    del blks
    log(f"design: X_proto {Xp.shape} X_band {Xb.shape} positives {int(y.sum())}")
    fold = np.full(len(y), -1, np.int8)
    for k, (_a, b) in enumerate(GroupKFold(cfg["cv"]["n_folds"]).split(Xp, y, sid)):
        fold[b] = k
    K = cfg["cv"]["n_folds"]
    folds = {k: sorted(str(x) for x in set(sid[fold == k])) for k in range(K)}
    log(f"folds: {folds}")

    # 2-4. HGB
    oof_hgb = None
    if cfg["hgb"]["enabled"]:
        t0 = time.time()
        oof_hgb = np.zeros(len(y))
        for k in range(K):
            m = fit_hgb(Xp[fold != k], y[fold != k], cfg)
            oof_hgb[fold == k] = m.predict_proba(Xp[fold == k])[:, 1]
        hgb = fit_hgb(Xp, y, cfg)
        with open(out / "hgb.pkl", "wb") as f:
            pickle.dump(hgb, f)
        timings["hgb_s"] = round(time.time() - t0, 1)
        log(f"hgb: OOF + final done ({timings['hgb_s']} s, n_iter final {hgb.n_iter_})")

    # TCN bag
    oof_tcn = None
    if cfg["tcn"]["enabled"]:
        t0 = time.time()
        tc = cfg["tcn"]
        X = np.hstack([Xp.astype(np.float32), Xb]) if cfg["features"]["use_bands"] else Xp.astype(np.float32)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = standardise(X, mu, sd, tc["clip_z"])
        del X
        np.savez(out / "tcn_norm.npz", mu=mu, sd=sd)
        probs = np.zeros((tc["seeds"], len(y)), np.float32)
        for k in range(K):
            tr = seqs(Xs, y, sid, fold != k)
            ho_ids = np.unique(sid[fold == k])
            for s in range(tc["seeds"]):
                model = train_tcn(tr, cfg, seed=tc["cv_seed_base"] * s + k, epochs=tc["epochs"])
                for h in ho_ids:
                    m = sid == h
                    probs[s, m] = predict_seq(model, Xs[m])
            log(f"tcn: CV fold {k} done ({time.time() - t0:.0f} s)")
        oof_tcn = probs.mean(0)
        del probs
        all_seqs = seqs(Xs, y, sid, np.ones(len(y), bool))
        for s in range(tc["seeds"]):
            model = train_tcn(all_seqs, cfg, seed=tc["final_seed_base"] + s, epochs=tc["epochs"])
            torch.save(model.state_dict(), out / f"tcn_{s}.pt")
        timings["tcn_s"] = round(time.time() - t0, 1)
        log(f"tcn: CV + final bag done ({timings['tcn_s']} s)")

    # operating point on the blended OOF
    t0 = time.time()
    oof = _blend(cfg, oof_tcn, oof_hgb)
    durations = {s.sid: s.duration for s in sources}
    gt = gt_records(ev, ignore_flagged=True)
    probs_by = {s: oof[sid == s] for s in pd.unique(sid)}
    oof_rank, g, table = tune(probs_by, gt, durations, cfg, log)
    timings["decode_tune_s"] = round(time.time() - t0, 1)
    cand = decode_streams(probs_by, g)
    oof_report = {"detector": summary(score_inproc(cand, gt, durations))}
    np.savez_compressed(out / "oof.npz", sid=sid, y=y, fold=fold,
                        oof_hgb=oof_hgb if oof_hgb is not None else np.zeros(0),
                        oof_tcn=oof_tcn if oof_tcn is not None else np.zeros(0), oof_blend=oof)
    cand.to_csv(out / "oof_candidates.csv", index=False)
    if cfg["report"]["oof_replay"] and cfg["boundary"]["method"] == "replay":
        t0 = time.time()
        final = _boundaries(sources, cand, cfg, durations, log)
        final.to_csv(out / "oof_final.csv", index=False)
        oof_report["final"] = summary(score_inproc(final, gt, durations))
        timings["oof_replay_s"] = round(time.time() - t0, 1)
    log(f"train OOF (in-process official evaluator): {json.dumps(oof_report)}")

    meta = {
        "uwbfall_version": __version__, "git_rev": _git_rev(), "extractor_version": EXTRACTOR_VERSION,
        "config_hash": config_hash(cfg), "train_streams": [s.sid for s in sources], "content_keys": keys,
        "folds": folds, "proto_cols": cols, "n_blocks": int(len(y)),
        "postproc": list(g), "oof_rank_f1_tuned": oof_rank, "oof": oof_report,
        "decode_grid_table": table, "timings": {**timings, "total_s": round(time.time() - t_start, 1)},
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        "env": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
    }
    with open(out / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log(f"fit done in {meta['timings']['total_s']} s -> {out}")
    return out


# ============================================================================ predict
def load_bundle(path) -> dict:
    p = Path(path)
    cfg = json.loads((p / "config.json").read_text())
    meta = json.loads((p / "meta.json").read_text())
    B = {"cfg": cfg, "meta": meta, "path": p}
    with open(p / "pca.pkl", "rb") as f:
        B["pca"] = pickle.load(f)
    B["hgb"] = None
    if cfg["hgb"]["enabled"]:
        with open(p / "hgb.pkl", "rb") as f:
            B["hgb"] = pickle.load(f)
    B["tcn"] = []
    if cfg["tcn"]["enabled"]:
        z = np.load(p / "tcn_norm.npz")
        B["mu"], B["sd"] = z["mu"], z["sd"]
        cin = len(B["mu"])
        for s in range(cfg["tcn"]["seeds"]):
            m = build_tcn(cin, cfg)
            m.load_state_dict(torch.load(p / f"tcn_{s}.pt", weights_only=True))
            m.eval()
            B["tcn"].append(m)
    return B


def stream_probs(B: dict, F: dict) -> tuple[np.ndarray, dict]:
    cfg = B["cfg"]
    blk = stream_blocks(F, cfg)
    xp, xb, _ = design(blk, B["pca"], cfg)
    p_hgb = B["hgb"].predict_proba(xp)[:, 1] if B["hgb"] is not None else None
    p_tcn = None
    if B["tcn"]:
        X = np.hstack([xp.astype(np.float32), xb]) if cfg["features"]["use_bands"] else xp.astype(np.float32)
        Xs = standardise(X, B["mu"], B["sd"], cfg["tcn"]["clip_z"])
        p_tcn = np.mean([predict_seq(m, Xs) for m in B["tcn"]], 0)
    return _blend(cfg, p_tcn, p_hgb), {"hgb": p_hgb, "tcn": p_tcn}


def _boundaries(sources, cand: pd.DataFrame, cfg: dict, durations: dict, log) -> pd.DataFrame:
    """Replay boundaries for candidate rows of several sources (one source in memory at a time)."""
    out = []
    for s in sorted(sources, key=lambda s: s.sid):
        c = cand[cand.stream_id == s.sid]
        if not len(c):
            continue
        F = get_features(s, cfg)
        D = replay_stream(s, F["prof_02"], c.reset_index(drop=True), cfg)
        s.release()
        a, e, _use = refine(D, cfg["boundary"])
        out.append(to_preds(D, a, e, durations))
    return pd.concat(out, ignore_index=True) if out else cand.iloc[:0]


def predict(bundle, sources: list[StreamSource], log=None, return_debug: bool = False):
    B = bundle if isinstance(bundle, dict) else load_bundle(bundle)
    cfg = B["cfg"]
    check_implemented(cfg)
    _lim = setup_torch(cfg)  # noqa: F841
    log = log or _logger(None)
    g = tuple(B["meta"]["postproc"])
    rows, cands, dbg = [], [], {}
    durations = {s.sid: s.duration for s in sources}
    for s in sorted(sources, key=lambda s: s.sid):
        t0 = time.time()
        F = get_features(s, cfg, log)
        p, parts = stream_probs(B, F)
        c = events_df(to_events(s.sid, p, *g))
        if cfg["boundary"]["method"] == "replay" and len(c):
            D = replay_stream(s, F["prof_02"], c, cfg)
            a, e, use = refine(D, cfg["boundary"])
            r = to_preds(D, a, e, durations)
            n_rep = int(use.sum())
        else:
            r = to_preds(c, c.t_start.values, c.t_end.values, durations) if len(c) else c
            n_rep = 0
        s.release()
        rows.append(r)
        cands.append(c)
        if return_debug:
            dbg[s.sid] = {"p": p, **parts}
        log(f"predict {s.sid}: {len(c)} candidates, {n_rep} replay-refined, {time.time() - t0:.1f} s")
    cols = ["stream_id", "t_start", "t_end", "score", "t_alarm"]
    pred = pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)
    if return_debug:
        return pred, pd.concat(cands, ignore_index=True), dbg
    return pred

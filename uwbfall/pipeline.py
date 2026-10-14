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
from . import clip as CL
from . import hitmodel as HM
from .boundary import refine, replay_stream, to_preds
from .config import check_implemented, config_hash, load_config
from .decode import decode_streams, events_df, to_events, tune
from .extract import EXTRACTOR_VERSION, get_features
from .features import block_labels, design, fit_pca, hgb_design, ignore_clip_mask, stream_blocks
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
    ign = cfg["labels"]["ignore_clips"] == "zero_weight"
    need_comp = cfg["hgb"]["postfall_class"] or cfg["clip"]["segclf"]["enabled"] or ign
    comps = {s.sid: CL.composition_of(s) for s in sources} if need_comp else {}
    Y3, clip_in, Xh, W = [], [], [], []
    for s, b in zip(sources, blks):
        xp, xb, cols = design(b, pca, cfg)
        y, _fl = block_labels(ev, s.sid, b["n"])
        Xp.append(xp); Xb.append(xb); Y.append(y.astype(np.int8)); SID.append(np.full(b["n"], s.sid))
        Xh.append(hgb_design(b, xp, cfg))
        if ign:
            W.append(np.where(ignore_clip_mask(ev, comps[s.sid], s.sid, b["n"]), 0.0, 1.0))
        if cfg["hgb"]["postfall_class"]:
            pf = CL.postfall_labels(ev, comps[s.sid], s.sid, b["n"])
            Y3.append(np.where(y, 1, np.where(pf, 2, 0)).astype(np.int8))
        if cfg["clip"]["segclf"]["enabled"] or cfg["decode"]["one_event_per_clip"] == "segment":
            fpc = [i for i, c in enumerate(cols) if c.startswith("fp") and not c.startswith("fpseg")]
            clip_in.append((s, b["base"], xp[:, fpc]))
        b.pop("amp")
    Xp, Xb, y, sid = np.vstack(Xp), np.vstack(Xb), np.concatenate(Y), np.concatenate(SID)
    Xh = np.vstack(Xh)
    W = np.concatenate(W) if W else None
    w_hgb = W if W is not None and "hgb" in cfg["labels"]["ignore_members"] else None
    w_tcn = W if W is not None and "tcn" in cfg["labels"]["ignore_members"] else None
    if W is not None:
        log(f"labels: ignore_clips zero_weight on {int((W == 0).sum())} blocks ({(W == 0).mean():.3%})")
    y_hgb = np.concatenate(Y3) if Y3 else y
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
            m = fit_hgb(Xh[fold != k], y_hgb[fold != k], cfg, None if w_hgb is None else w_hgb[fold != k])
            oof_hgb[fold == k] = m.predict_proba(Xh[fold == k])[:, 1]  # class 1 = fall (also with post-fall class 2)
        hgb = fit_hgb(Xh, y_hgb, cfg, w_hgb)
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
            tr = seqs(Xs, y, sid, fold != k, w_tcn)
            ho_ids = np.unique(sid[fold == k])
            for s in range(tc["seeds"]):
                model = train_tcn(tr, cfg, seed=tc["cv_seed_base"] * s + k + tc["seed_offset"], epochs=tc["epochs"])
                for h in ho_ids:
                    m = sid == h
                    probs[s, m] = predict_seq(model, Xs[m])
            log(f"tcn: CV fold {k} done ({time.time() - t0:.0f} s)")
        oof_tcn = probs.mean(0)
        del probs
        all_seqs = seqs(Xs, y, sid, np.ones(len(y), bool), w_tcn)
        for s in range(tc["seeds"]):
            model = train_tcn(all_seqs, cfg, seed=tc["final_seed_base"] + s + tc["seed_offset"], epochs=tc["epochs"])
            torch.save(model.state_dict(), out / f"tcn_{s}.pt")
        timings["tcn_s"] = round(time.time() - t0, 1)
        log(f"tcn: CV + final bag done ({timings['tcn_s']} s)")

    base_meta = {"train_streams": [s.sid for s in sources], "content_keys": keys, "folds": folds,
                 "proto_cols": cols, "n_blocks": int(len(y))}
    return _finish(out, cfg, log, sources, ev, sid, y, fold, oof_hgb, oof_tcn, clip_in, comps, pca, base_meta,
                   timings, t_start)


def _finish(out, cfg, log, sources, ev, sid, y, fold, oof_hgb, oof_tcn, clip_in, comps, pca, base_meta, timings,
            t_start) -> Path:
    """Post-model stages shared by fit() and retune(): clip stage, blended (gated) OOF -> operating point,
    OOF replay + final NMS, bundle metadata."""
    # clip stage (W3): junction threshold, segment classifier, subject-grouped OOF gate
    clip_state, gate_oof = None, None
    if clip_in:
        t0 = time.time()
        clip_state, gate_oof = _fit_clip(clip_in, comps, cfg, log)
        with open(out / "clip.pkl", "wb") as f:
            pickle.dump(clip_state, f)
        timings["clip_s"] = round(time.time() - t0, 1)
    clip_in = None

    # operating point on the blended OOF
    t0 = time.time()
    oof = _blend(cfg, oof_tcn, oof_hgb)
    if cfg["clip"]["gate"] != "none":
        oof = CL.gate_probs(oof, np.concatenate([gate_oof[x] for x in pd.unique(sid)]), cfg["clip"]["gate"],
                            cfg["clip"]["gate_floor"])
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
        final = _nms(final, cfg, clip_state["peaks"] if clip_state else {})
        final.to_csv(out / "oof_final.csv", index=False)
        oof_report["final"] = summary(score_inproc(final, gt, durations))
        timings["oof_replay_s"] = round(time.time() - t0, 1)
    log(f"train OOF (in-process official evaluator): {json.dumps(oof_report)}")

    meta = {
        "uwbfall_version": __version__, "git_rev": _git_rev(), "extractor_version": EXTRACTOR_VERSION,
        "config_hash": config_hash(cfg), **base_meta,
        "postproc": list(g), "oof_rank_f1_tuned": oof_rank, "oof": oof_report,
        "decode_grid_table": table,
        "clip": None if clip_state is None else {"junction_thr": clip_state["thr"], "junction_diag": clip_state["diag"],
                                                 "segclf_features": clip_state["features"]}, "timings": {**timings, "total_s": round(time.time() - t_start, 1)},
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


def _fit_clip(clip_in, comps, cfg, log):
    """Junction threshold (fall-junction recall on the training streams), segment tables, segment classifier.
    Returns (state for the bundle, {sid: per-block OOF P(fall clip)} grouped by segment subject)."""
    from sklearn.model_selection import GroupKFold
    c = cfg["clip"]
    stats = {}
    for s, _base, _fp in clip_in:
        stats[s.sid] = CL.junction_stat(get_features(s, cfg)["prof_02"], c["junction_W"])
    thr = CL.junction_threshold(stats, {k: comps[k] for k in stats}, c["junction_recall"])
    diag = CL.junction_recall(stats, {k: comps[k] for k in stats}, thr, c["junction_min_sep_s"])
    log(f"clip: junction threshold {thr:.4f}, train diag {json.dumps(diag)}")
    state = {"thr": thr, "diag": diag, "peaks": {}, "segclf": None, "features": None}
    gate_oof = None
    if c["segclf"]["enabled"]:
        tabs = []
        for s, base, fp in clip_in:
            F = get_features(s, cfg)
            segs, tab, pk_t, _ = CL.stream_segments(F, base, fp, thr, cfg)
            s.release()
            tab["sid"], tab["a"], tab["b"] = s.sid, segs[:, 0], segs[:, 1]
            tab["y"] = CL.segment_labels(segs, comps[s.sid])
            tab["subj"] = CL.segment_subjects(segs, comps[s.sid])
            state["peaks"][s.sid] = pk_t
            tabs.append(tab)
        T = pd.concat(tabs, ignore_index=True)
        feats = [x for x in T.columns if x not in ("sid", "a", "b", "y", "subj")]
        fs = CL.feature_set(feats, c["segclf"]["features"])
        oof = np.zeros(len(T))
        groups = T.subj if c["segclf"]["oof_groups"] == "subject" else T.sid
        for a, b in GroupKFold(4).split(T, T.y, groups):
            m = CL.lgbm_segclf(cfg).fit(T.iloc[a][fs], T.y.iloc[a])
            oof[b] = m.predict_proba(T.iloc[b][fs])[:, 1]
        state["segclf"] = CL.lgbm_segclf(cfg).fit(T[fs], T.y)
        state["features"] = fs
        gate_oof = {}
        for sid_, g in T.assign(p=oof).groupby("sid", sort=False):
            n = len(stats[sid_])
            gate_oof[sid_] = CL.block_gate(np.stack([g.a.values, g.b.values], 1), g.p.values, n)
        from sklearn.metrics import roc_auc_score
        log(f"clip: segment classifier on {len(T)} segments ({len(fs)} features), subject-grouped OOF AUC "
            f"fall-vs-rest {roc_auc_score(T.y == 1, oof):.4f}")
    else:
        for s, _base, _fp in clip_in:
            state["peaks"][s.sid] = CL.junction_peaks(stats[s.sid], thr, c["junction_min_sep_s"]) * CL.STEP
    return state, gate_oof


RETUNE_SECTIONS = ("clip", "decode", "boundary", "report")
MODEL_FILES = ("pca.pkl", "hgb.pkl", "tcn_norm.npz", "tcn_0.pt", "tcn_1.pt", "tcn_2.pt", "tcn_3.pt", "tcn_4.pt")


def retune(bundle, train_sources: list[StreamSource], events: pd.DataFrame | None, overrides: dict, out_dir,
           log=None) -> Path:
    """Re-run the post-model stages of fit() on an existing bundle without retraining HGB / TCN: clip stage
    (junction threshold + segment classifier), gated OOF -> operating point, OOF replay + final NMS.

    `overrides` may only touch the sections in RETUNE_SECTIONS; the model sections stay those of the bundle.
    The training sources must be the bundle's (checked by stream id and block count). The result equals what
    fit() would produce with the merged config, given the same HGB / TCN OOF probabilities and final models."""
    import shutil
    src = Path(bundle)
    bad = set(overrides) - set(RETUNE_SECTIONS)
    if bad:
        raise ValueError(f"retune can only change {RETUNE_SECTIONS}, not {sorted(bad)}")
    raw = json.loads((src / "config.json").read_text())
    if "clip" not in raw:
        raw["clip"] = {"segclf": {"enabled": False}, "gate": "none"}
    base = load_config(raw)
    cfg = load_config(_deep(base, overrides))
    check_implemented(cfg)
    if cfg["hgb"]["postfall_class"] != base["hgb"]["postfall_class"]:
        raise ValueError("hgb.postfall_class needs a refit")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or _logger(out / "train.log")
    _lim = setup_torch(cfg)  # noqa: F841
    t_start = time.time()
    for f in MODEL_FILES:
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    meta0 = json.loads((src / "meta.json").read_text())
    with open(src / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    z = np.load(src / "oof.npz")
    sid, y, fold = z["sid"], z["y"], z["fold"]
    oof_hgb = z["oof_hgb"] if len(z["oof_hgb"]) else None
    oof_tcn = z["oof_tcn"] if len(z["oof_tcn"]) else None
    sources = sorted(train_sources, key=lambda s: s.sid)
    if [s.sid for s in sources] != meta0["train_streams"]:
        raise ValueError("retune: training sources differ from the bundle's")
    ev = events_for(sources, events)
    log(f"retune {src} -> {out}: overrides {json.dumps(overrides)}, config {config_hash(cfg)}")
    need_clip = cfg["clip"]["segclf"]["enabled"] or cfg["decode"]["one_event_per_clip"] == "segment"
    comps = {s.sid: CL.composition_of(s) for s in sources} if need_clip else {}
    clip_in = []
    if need_clip:
        for s in sources:
            F = get_features(s, cfg, log)
            b = stream_blocks(F, cfg)
            xp, _xb, cols = design(b, pca, cfg)
            if b["n"] != int((sid == s.sid).sum()):
                raise ValueError(f"retune: block count of {s.sid} differs from the bundle OOF")
            fpc = [i for i, c in enumerate(cols) if c.startswith("fp") and not c.startswith("fpseg")]
            clip_in.append((s, b["base"], xp[:, fpc]))
            s.release()
            del F
    base_meta = {k: meta0[k] for k in ("train_streams", "content_keys", "folds", "proto_cols", "n_blocks")}
    base_meta["retuned_from"] = {"path": str(src), "config_hash": meta0.get("config_hash"),
                                 "git_rev": meta0.get("git_rev")}
    timings = {k: v for k, v in meta0.get("timings", {}).items() if k in ("features_s", "hgb_s", "tcn_s")}
    return _finish(out, cfg, log, sources, ev, sid, y, fold, oof_hgb, oof_tcn, clip_in, comps, pca, base_meta,
                   timings, t_start)


def attach_hit(bundle, hit_state: dict, out_dir) -> Path:
    """W5 (I3): copy a bundle and attach an event hit model. hit_state = {"model": hitmodel.HitModel,
    "tau": float | None, "meta": {...}} (from hitmodel.fit_nested on SR-CV OOF candidate tables). Sets
    score.hit_model="lgbm" and score.threshold=tau in the copied config.json."""
    import shutil
    src, out = Path(bundle), Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    raw = json.loads((out / "config.json").read_text())
    cfg = load_config(raw if "clip" in raw else {**raw, "clip": {"segclf": {"enabled": False}, "gate": "none"}})
    cfg["score"].update({"hit_model": "lgbm", "threshold": hit_state.get("tau"),
                         **{k: v for k, v in hit_state["model"].cfg_s.items() if k in cfg["score"] and k not in
                            ("hit_model", "threshold")}})
    check_implemented(cfg)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    with open(out / "hit.pkl", "wb") as f:
        pickle.dump(hit_state, f)
    meta = json.loads((out / "meta.json").read_text())
    meta["hit_model"] = hit_state.get("meta", {})
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return out


def _deep(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) and k != "grid" else v
    return out


def _nms(df: pd.DataFrame, cfg: dict, peaks: dict) -> pd.DataFrame:
    mode = cfg["decode"]["one_event_per_clip"]
    if mode == "time":
        return CL.nms_time(df, cfg["decode"]["nms_gap_s"]).reset_index(drop=True)
    if mode == "segment":
        return CL.nms_segment(df, peaks, cfg["decode"]["nms_max_span_s"]).reset_index(drop=True)
    return df


# ============================================================================ predict
def load_bundle(path) -> dict:
    p = Path(path)
    raw = json.loads((p / "config.json").read_text())
    if "clip" not in raw:  # bundle trained before the clip stage existed: keep it OFF (no clip.pkl, untuned op)
        raw["clip"] = {"segclf": {"enabled": False}, "gate": "none"}
    cfg = load_config(raw)
    meta = json.loads((p / "meta.json").read_text())
    B = {"cfg": cfg, "meta": meta, "path": p}
    with open(p / "pca.pkl", "rb") as f:
        B["pca"] = pickle.load(f)
    B["hgb"] = None
    if cfg["hgb"]["enabled"]:
        with open(p / "hgb.pkl", "rb") as f:
            B["hgb"] = pickle.load(f)
    B["hit"] = None
    if cfg["score"]["hit_model"] is not None:
        with open(p / "hit.pkl", "rb") as f:
            B["hit"] = pickle.load(f)
    B["clip"] = None
    if (p / "clip.pkl").exists():
        with open(p / "clip.pkl", "rb") as f:
            B["clip"] = pickle.load(f)
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
    xp, xb, cols = design(blk, B["pca"], cfg)
    p_hgb = B["hgb"].predict_proba(hgb_design(blk, xp, cfg))[:, 1] if B["hgb"] is not None else None
    p_tcn = None
    if B["tcn"]:
        X = np.hstack([xp.astype(np.float32), xb]) if cfg["features"]["use_bands"] else xp.astype(np.float32)
        Xs = standardise(X, B["mu"], B["sd"], cfg["tcn"]["clip_z"])
        p_tcn = np.mean([predict_seq(m, Xs) for m in B["tcn"]], 0)
    p = _blend(cfg, p_tcn, p_hgb)
    parts = {"hgb": p_hgb, "tcn": p_tcn, "peaks": None}
    if B.get("clip") is not None:
        fpc = [i for i, c in enumerate(cols) if c.startswith("fp") and not c.startswith("fpseg")]
        segs, tab, pk_t, _ = CL.stream_segments(F, blk["base"], xp[:, fpc], B["clip"]["thr"], cfg)
        parts["peaks"] = pk_t
        if B["clip"]["segclf"] is not None and cfg["clip"]["gate"] != "none":
            pf = B["clip"]["segclf"].predict_proba(tab[B["clip"]["features"]])[:, 1]
            parts["gate"] = CL.block_gate(segs, pf, blk["n"])
            p = CL.gate_probs(p, parts["gate"], cfg["clip"]["gate"], cfg["clip"]["gate_floor"])
    return p, parts


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
            if B.get("hit") is not None:   # W5 (I3): event hit model replaces the score before the final NMS
                X = HM.candidate_features(D, cfg["boundary"], {s.sid: p}, {s.sid: parts},
                                          {s.sid: parts["peaks"]} if parts.get("peaks") is not None else {})
                r = HM.final_rows(D, cfg["boundary"], durations,
                                  score=np.round(B["hit"]["model"].predict(X), 6)).drop(columns="row")
            else:
                r = to_preds(D, a, e, durations)
            n_rep = int(use.sum())
        else:
            r = to_preds(c, c.t_start.values, c.t_end.values, durations) if len(c) else c
            n_rep = 0
        s.release()
        if len(r):
            r = _nms(r, cfg, {s.sid: parts["peaks"]} if parts.get("peaks") is not None else {})
        if len(r) and B.get("hit") is not None and cfg["score"]["threshold"] is not None:
            r = r[r.score >= float(cfg["score"]["threshold"])].reset_index(drop=True)
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

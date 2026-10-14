"""W5 (I3): event-level hit model for the score column, and the operating threshold on it.

Each candidate event (detector interval + annotator-replay internals + refined bounds) gets features from the
frame posteriors, the replay crops, the segment gate and the junctions. Three LightGBM classifiers predict
hit@theta (theta 0.1 / 0.3 / 0.5: tIoU of the refined interval with an evaluable ground-truth fall >= theta);
the event score is their mean = expected mean hit over the ranking thetas.

Training rows are the training-OOF candidates of a fit (the same candidates the operating point is tuned on).
Candidates that overlap an ignore zone (official rule: +-2 s around a flagged event) and hit no evaluable fall
are FREE for the evaluator (neither TP nor FP), so they are left out of the hit-model training instead of being
taught as negatives (`hit_labels` -> mask `train_ok`).

Operating threshold (`choose_threshold`): rows with hit score < tau are dropped after the final NMS. tau is chosen
on the training OOF with *nested* hit scores (inner subject-grouped CV over training candidates, so the scores the
threshold is tuned on never come from a model that saw those candidates), maximising the official ranking F1
(mean F1 over theta .1/.3/.5) with the in-process official evaluator.

Config (section "score"): hit_model None | "lgbm"; threshold None | "nested"; see config.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .boundary import refine, tiou, to_preds
from .paths import STEP_S

THETAS = (0.1, 0.3, 0.5)
HIT_FEATURES = [
    # detector / posterior
    "score", "det_dur", "p_max", "p_mean", "p_frac_hi", "tcn_max", "tcn_mean", "hgb_max", "hgb_mean",
    # segment classifier (P(fall clip)) and junctions
    "gate_mean", "gate_max", "to_jprev", "to_jnext", "seg_len", "to_pk_prev", "to_pk_next",
    # replay
    "use", "n_ok", "n_fin", "t_ref_det", "ref_dur", "ds", "de", "spread_s", "spread_e", "mean_t_med",
    "lowq_frac", "rev_frac", "n_rev_mean", "anchor_mode_frac", "onset_spread_mean", "pk_rel",
    # local context (other candidates of the same stream)
    "n_near", "score_rank_near", "score_gap_near",
]


def _crop_arrays(D: pd.DataFrame, crops):
    S = np.vstack([D[f"{c}_s"].values for c in crops]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in crops]).astype(float)
    return S, E


def candidate_features(D: pd.DataFrame, b: dict, probs: dict, parts: dict | None = None, peaks: dict | None = None,
                       near_s: float = 4.0) -> pd.DataFrame:
    """Feature table (HIT_FEATURES) for candidate rows D (columns stream_id, t_start, t_end, score + replay columns
    from boundary.replay_stream). probs[sid] = gated blend posterior per 0.2 s block; parts[sid] = dict with optional
    'tcn', 'hgb' (ungated member posteriors) and 'gate' (block P(fall clip)); peaks[sid] = clip-stage junction
    times (s). Missing inputs give NaN features (LightGBM handles them)."""
    parts = parts or {}
    peaks = peaks or {}
    crops = b["crops"]
    s_ref, e_ref, use = refine(D, b)
    s0, e0 = D.t_start.values.astype(float), D.t_end.values.astype(float)
    S, E = _crop_arrays(D, crops)
    fin = np.isfinite(S) & np.isfinite(E) & (E > S)
    ok = fin & (tiou(S, E, s0, e0) >= b["gate"])
    nan = np.full(len(D), np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sp_s = np.nanstd(np.where(fin, S, np.nan), 0)
        sp_e = np.nanstd(np.where(fin, E, np.nan), 0)
        t_med = np.where(fin, tiou(S, E, s_ref[None], e_ref[None]), np.nan)
        mean_t_med = np.nanmean(t_med, 0)
        lowq = np.vstack([D[f"{c}_lowq"].astype(float).values if f"{c}_lowq" in D else nan for c in crops])
        rev = np.vstack([D[f"{c}_rev"].fillna("").astype(str).values if f"{c}_rev" in D else np.array([""] * len(D))
                         for c in crops])
        nrev = np.vectorize(lambda x: float(len([y for y in x.split(";") if y and y != "nan"])))(rev) if len(D) else rev
        nrev = np.where(fin, nrev, np.nan)
        lowq_frac = np.nanmean(np.where(fin, lowq, np.nan), 0)
        rev_frac = np.nanmean(np.where(fin, (nrev > 0).astype(float), np.nan), 0)
        n_rev_mean = np.nanmean(nrev, 0)
        anc = np.vstack([D[f"{c}_anchor"].astype(float).values if f"{c}_anchor" in D else nan for c in crops])
        onset = np.vstack([D[f"{c}_spread"].astype(float).values if f"{c}_spread" in D else nan for c in crops])
        onset_mean = np.nanmean(np.where(fin, onset, np.nan), 0)
        pk = np.vstack([D[f"{c}_pk"].astype(float).values if f"{c}_pk" in D else nan for c in crops])
        pk_med = np.nanmedian(np.where(ok, pk, np.nan), 0)
    anchor_mode = np.full(len(D), np.nan)
    for i in range(len(D)):
        a = anc[fin[:, i], i]
        a = a[np.isfinite(a)]
        if len(a):
            anchor_mode[i] = np.bincount(a.astype(int)).max() / len(a)
    rows = []
    for i, r in enumerate(D.itertuples(index=False)):
        sid = r.stream_id
        p = probs.get(sid)
        pp = parts.get(sid, {})
        a, z = int(round(r.t_start / STEP_S)), max(int(round(r.t_end / STEP_S)), int(round(r.t_start / STEP_S)) + 1)

        def seg(x):
            return None if x is None else np.asarray(x)[a:z]
        f = {}
        for nm, arr in (("p", seg(p)), ("tcn", seg(pp.get("tcn"))), ("hgb", seg(pp.get("hgb")))):
            ok_ = arr is not None and len(arr) > 0
            f[f"{nm}_max"] = float(arr.max()) if ok_ else np.nan
            f[f"{nm}_mean"] = float(arr.mean()) if ok_ else np.nan
        pa = seg(p)
        f["p_frac_hi"] = float((pa >= 0.5).mean()) if pa is not None and len(pa) else np.nan
        g = seg(pp.get("gate"))
        f["gate_mean"] = float(g.mean()) if g is not None and len(g) else np.nan
        f["gate_max"] = float(g.max()) if g is not None and len(g) else np.nan
        pk_t = peaks.get(sid)
        if pk_t is not None and len(pk_t):
            pk_t = np.asarray(pk_t, float)
            prv, nxt = pk_t[pk_t <= s_ref[i]], pk_t[pk_t >= e_ref[i]]
            f["to_pk_prev"] = s_ref[i] - prv[-1] if len(prv) else np.nan
            f["to_pk_next"] = nxt[0] - e_ref[i] if len(nxt) else np.nan
            f["seg_len"] = (nxt[0] if len(nxt) else np.nan) - (prv[-1] if len(prv) else np.nan)
        else:
            f["to_pk_prev"] = f["to_pk_next"] = f["seg_len"] = np.nan
        rows.append(f)
    X = pd.DataFrame(rows, index=D.index)
    X["score"] = D.score.values.astype(float)
    X["det_dur"] = e0 - s0
    X["to_jprev"] = s_ref - D.jprev.values.astype(float) if "jprev" in D else nan
    X["to_jnext"] = D.jnext.values.astype(float) - e_ref if "jnext" in D else nan
    X["use"] = use.astype(float)
    X["n_ok"] = ok.sum(0).astype(float)
    X["n_fin"] = fin.sum(0).astype(float)
    X["t_ref_det"] = tiou(s_ref, e_ref, s0, e0)
    X["ref_dur"] = e_ref - s_ref
    X["ds"], X["de"] = s_ref - s0, e_ref - e0
    X["spread_s"], X["spread_e"] = sp_s, sp_e
    X["mean_t_med"] = mean_t_med
    X["lowq_frac"], X["rev_frac"], X["n_rev_mean"] = lowq_frac, rev_frac, n_rev_mean
    X["anchor_mode_frac"] = anchor_mode
    X["onset_spread_mean"] = onset_mean
    X["pk_rel"] = (pk_med - s_ref) / np.maximum(e_ref - s_ref, 1e-3)
    # local context: candidates of the same stream within near_s s (gap between intervals)
    n_near, rank, gap = np.zeros(len(D)), np.zeros(len(D)), np.zeros(len(D))
    sc = D.score.values.astype(float)
    for _sid, idx in D.groupby("stream_id", sort=False).indices.items():
        for i in idx:
            d = np.maximum(s0[idx] - e0[i], s0[i] - e0[idx])
            nb = idx[(d < near_s) & (idx != i)]
            n_near[i] = len(nb)
            rank[i] = float((sc[nb] > sc[i]).sum())
            gap[i] = sc[i] - sc[nb].max() if len(nb) else 1.0
    X["n_near"], X["score_rank_near"], X["score_gap_near"] = n_near, rank, gap
    return X[HIT_FEATURES]


def hit_labels(sids, s, e, gt: list[dict], margin: float = 2.0) -> pd.DataFrame:
    """Per row: best tIoU with an evaluable GT fall of its stream, hit@theta, overlap with an ignore zone
    (+-margin s, official rule) and train_ok (= not a free ignore-zone row, i.e. hits something or overlaps no
    ignore zone)."""
    by = {}
    for g in gt:
        by.setdefault(g["stream_id"], []).append(g)
    best, ign = np.zeros(len(s)), np.zeros(len(s), bool)
    for i, (sid, a, z) in enumerate(zip(sids, s, e)):
        for g in by.get(sid, []):
            if g["ignore"]:
                if min(z, g["t_end"] + margin) > max(a, g["t_start"] - margin):
                    ign[i] = True
            else:
                best[i] = max(best[i], float(tiou(a, z, g["t_start"], g["t_end"])))
    out = pd.DataFrame({"best_iou": best, "ignore_overlap": ign})
    for th in THETAS:
        out[f"hit{th}"] = (best >= th - 1e-12).astype(int)
    out["train_ok"] = ~(ign & (best < THETAS[0]))
    return out


def _lgbm(cfg_s: dict, seed: int):
    import lightgbm as lgb
    p = cfg_s.get("lgbm", {})
    return lgb.LGBMClassifier(n_estimators=p.get("n_estimators", 200), learning_rate=p.get("learning_rate", 0.03),
                              num_leaves=p.get("num_leaves", 7), min_child_samples=p.get("min_child_samples", 20),
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1)


class HitModel:
    """Three classifiers (hit@0.1/0.3/0.5), each averaged over `seeds` LightGBM seeds; predict() = mean predicted
    hit probability over the thetas. A theta whose training labels are all one class uses that constant."""

    def __init__(self, cfg_s: dict, seed: int = 0):
        self.cfg_s, self.seed, self.models = cfg_s, seed, {}
        self.n_seeds = int(cfg_s.get("seeds", 3))

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        for th in THETAS:
            y = Y[f"hit{th}"].values
            if y.min() == y.max():
                self.models[th] = float(y[0])
            else:
                self.models[th] = [_lgbm(self.cfg_s, self.seed + k).fit(X[HIT_FEATURES], y)
                                   for k in range(self.n_seeds)]
        return self

    def predict_theta(self, X: pd.DataFrame) -> dict:
        out = {}
        for th, m in self.models.items():
            out[th] = (np.full(len(X), m) if isinstance(m, float) else
                       np.mean([q.predict_proba(X[HIT_FEATURES])[:, 1] for q in m], 0))
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not len(X):
            return np.zeros(0)
        return np.mean(list(self.predict_theta(X).values()), 0)

    def importance(self) -> dict:
        imp = np.zeros(len(HIT_FEATURES))
        for m in self.models.values():
            if not isinstance(m, float):
                imp += np.sum([q.booster_.feature_importance("gain") for q in m], 0)
        return dict(sorted(zip(HIT_FEATURES, (imp / max(imp.sum(), 1e-12)).round(4).tolist()), key=lambda x: -x[1]))


def inner_oof(X: pd.DataFrame, Y: pd.DataFrame, groups, cfg_s: dict, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Group-K-fold OOF hit scores on the training candidates (rows with train_ok=False are scored by models that
    never train on them either)."""
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    oof = np.full(len(X), np.nan)
    k = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(k).split(X, groups=groups):
        trm = tr[Y.train_ok.values[tr]]
        oof[te] = HitModel(cfg_s, seed).fit(X.iloc[trm], Y.iloc[trm]).predict(X.iloc[te])
    return oof


def final_rows(D: pd.DataFrame, b: dict, durations: dict, score: np.ndarray | None = None) -> pd.DataFrame:
    """Refined prediction rows (boundary.to_preds) keeping the candidate index (column 'row'); `score` replaces
    the detector score."""
    s, e, _ = refine(D, b)
    D2 = D.assign(score=score) if score is not None else D
    out = []
    for i, (sid, a, z, sc) in enumerate(zip(D2.stream_id, s, e, D2.score)):
        a = max(0.0, a); z = min(durations[sid], z)
        if z > a:
            out.append({"stream_id": sid, "t_start": round(float(a), 3), "t_end": round(float(z), 3),
                        "score": float(sc), "t_alarm": round(float(a), 3), "row": i})
    cols = ["stream_id", "t_start", "t_end", "score", "t_alarm", "row"]
    return pd.DataFrame(out, columns=cols)


def threshold_grid(cfg_s: dict) -> np.ndarray:
    g = cfg_s.get("tau_grid", [0.0, 0.6, 0.02])
    return np.round(np.arange(g[0], g[1] + 1e-9, g[2]), 4)


def choose_threshold(rows: pd.DataFrame, gt: list[dict], durations: dict, cfg_s: dict, nms) -> tuple[float, list]:
    """tau maximising the official ranking F1 of rows (scored by nested hit scores) after `nms` (callable on a
    prediction frame) and the cut score >= tau. The F1 curve is smoothed with a 3-point running mean before the
    argmax (ties -> smallest tau) to damp single-event jumps."""
    from .scoring import score_inproc
    post = nms(rows)
    taus = threshold_grid(cfg_s)
    f = []
    for tau in taus:
        r = post[post.score >= tau]
        f.append(score_inproc(r.drop(columns=[c for c in ("row",) if c in r]), gt, durations)["ranking_score"]["f1_mean"])
    f = np.array(f)
    sm = np.convolve(np.r_[f[0], f, f[-1]], np.ones(3) / 3, "valid")
    j = int(np.flatnonzero(sm >= sm.max() - 1e-12)[0])
    return float(taus[j]), [[float(t), float(x), float(y)] for t, x, y in zip(taus, f, sm)]


def fit_nested(parts: list[dict], gt: list[dict], durations: dict, cfg_s: dict, nms, groups=None, tau_parts=None,
               use_hit: bool = True):
    """Hit model + nested threshold from candidate tables. parts: list of {"X": features (HIT_FEATURES),
    "Y": hit_labels(...), "rows": final rows (stream_id, t_start, t_end, score, t_alarm)}, typically the
    held-out SR-CV test parts (one per fold and bag); groups: inner-CV group per part (the SR-CV fold, so rows of
    one fold never score themselves); tau_parts: indices of the parts whose rows the threshold is tuned on
    (default all). gt / durations cover the streams of the parts. With use_hit=False tau is tuned on the rows'
    own score (detector score) and no model is fitted. Returns (model | None, tau, curve, counts)."""
    X = pd.concat([p["X"] for p in parts], ignore_index=True)
    Y = pd.concat([p["Y"] for p in parts], ignore_index=True)
    rows = pd.concat([p["rows"] for p in parts], ignore_index=True)
    grp = np.concatenate([[g] * len(p["X"]) for g, p in zip(groups or range(len(parts)), parts)])
    tp = set(tau_parts if tau_parts is not None else range(len(parts)))
    sel = np.concatenate([[i in tp] * len(p["X"]) for i, p in enumerate(parts)])
    m = Y.train_ok.values
    model = HitModel(cfg_s).fit(X[m], Y[m]) if use_hit else None
    if use_hit:
        oof = inner_oof(X, Y, grp, cfg_s, n_splits=len(np.unique(grp)))
        rows = rows.assign(score=np.round(oof, 6))
    tau, curve = choose_threshold(rows[sel], gt, durations, cfg_s, nms)
    return model, tau, curve, {"n_rows": int(len(X)), "n_ok": int(m.sum()),
                               "neg@0.3": int((Y["hit0.3"].values[m] == 0).sum())}

"""SR-CV scoring with the official evaluator code, pooled OOF scoring and the PLAN v2 promotion rule.

All F1 numbers come from abc2027.evaluate (validate_predictions, load_ground_truth(ignore_flagged=True), evaluate,
_match_stream). Durations come from the composition (T/50, identical to the h5 the writer would produce).
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import core as C  # noqa: E402
from abc2027 import evaluate as E  # noqa: E402

RANK_THETAS = ("0.1", "0.3", "0.5")
ALL_THETAS = (0.1, 0.3, 0.5, 0.7)
N_FOLDS = 4


# ---------------------------------------------------------------- helpers
def _as_df(pred) -> pd.DataFrame:
    if isinstance(pred, pd.DataFrame):
        return pred
    return pd.read_csv(pred)


def _by_fold(preds) -> dict[int, pd.DataFrame]:
    """dict {fold: df|csv} or one DataFrame/CSV covering cv{k}_test_* streams -> {fold: df}."""
    if isinstance(preds, dict):
        return {int(k): _as_df(v) for k, v in preds.items()}
    df = _as_df(preds)
    return {k: df[df.stream_id.astype(str).str.startswith(f"cv{k}_test_")] for k in range(N_FOLDS)}


def _records(pred: pd.DataFrame, dur: dict[str, float]) -> list[dict]:
    """Official validate_predictions on the rows of ``pred`` whose stream belongs to ``dur``."""
    df = pred[pred.stream_id.astype(str).isin(dur)]
    cols = [c for c in ("stream_id", "t_start", "t_end", "score", "t_alarm") if c in df.columns]
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        if len(df):
            df[cols].to_csv(path, index=False)
            return E.validate_predictions(Path(path), dur)
        return []
    finally:
        os.unlink(path)


def _gt(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Official load_ground_truth(ignore_flagged=True) + the raw rows in the same order (for subjects)."""
    gt, raw = [], []
    for p in paths:
        gt += E.load_ground_truth(Path(p), ignore_flagged=True)
        raw += list(csv.DictReader(Path(p).open(encoding="utf-8", newline="")))
    return gt, raw


# ---------------------------------------------------------------- official scores
def score(pred, fold: int, part: str = "test") -> dict:
    """Official report (same dict as the CLI's --out json) on fold ``fold``'s streams of ``part``."""
    dur = C.durations(fold, part)
    gt, _ = _gt([C.events_csv(fold, part)])
    return E.evaluate(gt, _records(_as_df(pred), dur), dur, E.PRIMARY_THETA)


def score_pooled(preds, folds_=range(N_FOLDS)) -> dict:
    """Official report on the union of the folds' test streams (pooled OOF)."""
    by = _by_fold(preds)
    dur, recs, paths = {}, [], []
    for k in folds_:
        d = C.durations(k, "test")
        dur.update(d)
        recs += _records(by.get(k, pd.DataFrame(columns=["stream_id"])), d)
        paths.append(C.events_csv(k, "test"))
    gt, _ = _gt(paths)
    return E.evaluate(gt, recs, dur, E.PRIMARY_THETA)


def score_val(pred) -> dict:
    """Official report on the real val streams (durations from the h5, --ignore-flagged), = the CLI's json."""
    dur = E.stream_durations(C.PKG / "streams" / "val")
    gt = E.load_ground_truth(C.PKG / "streams_events.csv", ignore_flagged=True)
    return E.evaluate(gt, _records(_as_df(pred), dur), dur, E.PRIMARY_THETA)


def summary(rep: dict) -> dict:
    bt = rep["by_theta"]
    return {"rank_f1": rep["ranking_score"]["f1_mean"], **{f"f1@{t}": bt[t]["f1"] for t in ("0.1", "0.3", "0.5", "0.7")},
            "tp@0.3": bt["0.3"]["tp"], "fp@0.3": bt["0.3"]["fp"], "fn@0.3": bt["0.3"]["fn"],
            "start_med@0.3": bt["0.3"]["start_error_median_abs"], "fa_h@0.3": bt["0.3"]["false_alarms_per_hour"]}


# ---------------------------------------------------------------- subject-cluster attribution
def _segment_subject(comp: pd.DataFrame):
    by = {sid: (g.t_start.values, g.t_end.values, g.subject.values) for sid, g in comp.groupby("stream_id")}

    def find(sid: str, a: float, b: float) -> str:
        s, e, subj = by[sid]
        ov = np.minimum(e, b) - np.maximum(s, a)
        if ov.max() <= 0:  # zero-length overlap: use the midpoint
            m = 0.5 * (a + b)
            ov = -np.abs(0.5 * (s + e) - m)
        return str(subj[int(np.argmax(ov))])

    return find


def cluster_counts(preds, folds_=range(N_FOLDS)) -> pd.DataFrame:
    """Per (fold, subject, theta): tp, fp, fn under the official matching.

    TP/FN are attributed to the subject of the ground-truth event (its source recording). An unmatched,
    non-ignored prediction (FP) is attributed to the subject of the composition segment it overlaps most
    (fall subject for its Chute/Autre/Empty clips, ADL_n for activity pieces). Totals equal the official
    evaluate() counts (asserted).
    """
    by = _by_fold(preds)
    rows = []
    for k in folds_:
        dur = C.durations(k, "test")
        recs = _records(by.get(k, pd.DataFrame(columns=["stream_id"])), dur)
        gt, raw = _gt([C.events_csv(k, "test")])
        for g, r in zip(gt, raw):
            g["_subject"] = r["subject"]
        find = _segment_subject(C.composition(k, "test"))
        subjects = sorted(C.fold_subjects(k, "test"))
        acc = {(s, th): [0, 0, 0] for s in subjects for th in ALL_THETAS}
        official = E.evaluate(gt, recs, dur, E.PRIMARY_THETA)["by_theta"]
        for th in ALL_THETAS:
            for sid in sorted(dur):
                g_s = [g for g in gt if g["stream_id"] == sid]
                p_s = [p for p in recs if p["stream_id"] == sid]
                res = E._match_stream(g_s, p_s, th)
                scored = res["scored"]
                matched_gt = {gi for _pi, gi, _iou in res["pairs"]}
                matched_pred = {pi for pi, _gi, _iou in res["pairs"]}
                for gi, ev in enumerate(scored):
                    acc[(ev["_subject"], th)][0 if gi in matched_gt else 2] += 1
                ignored = [ev for ev in g_s if ev["ignore"]]
                for pi, p in enumerate(p_s):
                    if pi in matched_pred:
                        continue
                    if any(E._intervals_overlap(p["t_start"], p["t_end"], ev["t_start"] - E.IGNORE_MARGIN_S,
                                                ev["t_end"] + E.IGNORE_MARGIN_S) for ev in ignored):
                        continue
                    acc[(find(sid, p["t_start"], p["t_end"]), th)][1] += 1
            tot = np.sum([v for (s, t), v in acc.items() if t == th], axis=0)
            o = official[f"{th:.1f}"]
            assert tuple(tot) == (o["tp"], o["fp"], o["fn"]), (k, th, tot, o["tp"], o["fp"], o["fn"])
        for (s, th), (tp, fp, fn) in acc.items():
            rows.append({"fold": k, "subject": s, "stratum": "adl" if s.startswith("ADL_") else "fall",
                         "theta": f"{th:.1f}", "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def _count_tensor(cc: pd.DataFrame, subjects: list[str]) -> np.ndarray:
    """(n_subjects, 3 thetas, 3 counts) for the ranking thetas."""
    t = cc[cc.theta.isin(RANK_THETAS)].set_index(["subject", "theta"])[["tp", "fp", "fn"]]
    return np.stack([[t.loc[(s, th)].values.astype(float) for th in RANK_THETAS] for s in subjects])


def _rank_f1(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """W (B, n) cluster weights, X (n, 3, 3) -> ranking F1 (B,) = mean_theta 2TP / (2TP + FP + FN)."""
    S = np.einsum("bn,ntc->btc", W, X)
    tp, fp, fn = S[..., 0], S[..., 1], S[..., 2]
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return f1.mean(axis=1)


def _weights(strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Stratified cluster bootstrap weights: resample fall subjects and ADL subjects separately."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(strata)))
    for st in np.unique(strata):
        idx = np.flatnonzero(strata == st)
        draw = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        for j, i in enumerate(idx):
            W[:, i] = (draw == j).sum(axis=1)
    return W


def _arm(A) -> tuple[list[dict], np.ndarray, list[str], np.ndarray]:
    """A = {fold: pred} or list of such (seeds). Returns per-seed by-fold preds, mean count tensor, subjects, strata."""
    seeds = A if isinstance(A, (list, tuple)) else [A]
    seeds = [_by_fold(s) for s in seeds]
    ccs = [cluster_counts(s) for s in seeds]
    subjects = sorted(ccs[0].subject.unique())
    X = np.mean([_count_tensor(cc, subjects) for cc in ccs], axis=0)
    strata = np.array(["adl" if s.startswith("ADL_") else "fall" for s in subjects])
    return seeds, X, subjects, strata


def bootstrap_ci(A, n_boot: int = 5000, seed: int = 0) -> dict:
    """Pooled ranking F1 of one system (seed-averaged counts if a list) with a subject-cluster bootstrap 95% CI."""
    _seeds, X, subjects, strata = _arm(A)
    point = float(_rank_f1(np.ones((1, len(subjects))), X)[0])
    b = _rank_f1(_weights(strata, n_boot, seed), X)
    return {"rank_f1_pooled": point, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "boot_sd": float(b.std(ddof=1)), "n_clusters": len(subjects)}


def rank_f1_by_subject_mask(A, exclude: list[str]) -> float:
    """Pooled ranking F1 with the given subjects' clusters removed (diagnostic, e.g. exclude=['Test_P3_3'])."""
    _seeds, X, subjects, _strata = _arm(A)
    w = np.array([[0.0 if s in exclude else 1.0 for s in subjects]])
    return float(_rank_f1(w, X)[0])


def promote(A, B, n_boot: int = 5000, seed: int = 0, val_A=None, val_B=None, p_min: float = 0.90,
            min_folds: int = 3, val_tol: float = 0.02) -> dict:
    """PLAN v2 §2 promotion rule for candidate B against incumbent A.

    A, B: {fold: DataFrame|csv} (or one DataFrame covering cv{k}_test_* streams), or a LIST of those = seeds
          (counts are averaged over seeds, per-fold scores are seed means).
    val_A, val_B: optional official val ranking F1 (floats) or val prediction DataFrames/CSVs (scored here).
    Rule: P(delta pooled ranking F1 > 0) >= p_min under a paired, subject-stratified cluster bootstrap
          (24 fall subjects and 8 ADL subjects resampled separately, same draws for A and B), AND
          >= min_folds of 4 folds improve (official per-fold ranking F1, strict >), AND
          val_B >= val_A - val_tol (only checked if both are given).
    """
    sa, XA, subj_a, strata = _arm(A)
    sb, XB, subj_b, _ = _arm(B)
    assert subj_a == subj_b
    one = np.ones((1, len(subj_a)))
    pa, pb = float(_rank_f1(one, XA)[0]), float(_rank_f1(one, XB)[0])
    W = _weights(strata, n_boot, seed)
    ba, bb = _rank_f1(W, XA), _rank_f1(W, XB)
    d = bb - ba
    per_fold = []
    for k in range(N_FOLDS):
        fa = float(np.mean([score(s[k], k)["ranking_score"]["f1_mean"] for s in sa]))
        fb = float(np.mean([score(s[k], k)["ranking_score"]["f1_mean"] for s in sb]))
        per_fold.append({"fold": k, "A": fa, "B": fb, "delta": fb - fa})
    n_imp = sum(r["delta"] > 0 for r in per_fold)
    out = {
        "pooled_A": pa, "pooled_B": pb, "delta": pb - pa,
        "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_delta_gt_0": float((d > 0).mean()), "n_boot": n_boot, "n_clusters": len(subj_a),
        "n_seeds": [len(sa), len(sb)], "per_fold": per_fold, "n_folds_improved": int(n_imp),
    }
    ok = out["p_delta_gt_0"] >= p_min and n_imp >= min_folds
    if val_A is not None and val_B is not None:
        va = val_A if isinstance(val_A, float) else score_val(val_A)["ranking_score"]["f1_mean"]
        vb = val_B if isinstance(val_B, float) else score_val(val_B)["ranking_score"]["f1_mean"]
        out.update({"val_A": va, "val_B": vb, "val_ok": bool(vb >= va - val_tol)})
        ok = ok and out["val_ok"]
    else:
        out["val_ok"] = None
    out["promote"] = bool(ok)
    return out

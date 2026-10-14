"""End-to-end SR-CV baseline: the analysis/prototype_detector.py recipe ('full' features) on all 4 folds.

Per fold k:
  * features: analysis/prototype_detector.block_features/structure_features/add_fingerprint on the cached
    E1-extractor npz of the fold's regenerated streams (srcv.load_feats), PCA fingerprint fit on fold-train only;
  * labels: block centre inside any fold-train event (flagged included, as in the prototype);
  * HGB (same params, random_state=0), 4-fold GroupKFold-by-stream OOF on the fold's 13 training streams,
    80-point post-processing grid chosen on that OOF with the official evaluator (--ignore-flagged semantics);
  * final HGB on all fold-train streams -> predictions on the fold's held-out test streams.
Outputs srcv/results/baseline_prototype/fold_k.csv (+ summary json). Then pooled official score, per-fold
scores, subject-cluster bootstrap CI.

    python3 -m srcv.baseline_prototype [physics|full]
"""
from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/challenge1/analysis")
import prototype_detector as P  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from . import core as C  # noqa: E402
from . import scoring as S  # noqa: E402

STEP = P.STEP
GRID = [(thr, sm, md, gp) for thr in (0.3, 0.4, 0.5, 0.6, 0.7) for sm in (1, 3, 5) for md in (0.4, 0.8) for gp in (0.2, 0.6)]
PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
OUT = C.HERE / "results" / "baseline_prototype"


def build(srcs, ev: pd.DataFrame | None) -> list:
    """prototype_detector.build() for SR-CV sources (labels from the fold events instead of the package CSV)."""
    frames = []
    saved = P.FEATS
    try:
        for s in srcs:
            C.load_feats(s, compute=True)  # make sure the cache exists (and is not stale)
            P.FEATS = C.FEATS
            f, amp = P.block_features(s.stream_id)
            st, prev, nxt = P.structure_features(f)
            n = len(f["jump_max"])
            t = (np.arange(n) + 0.5) * STEP
            y = np.zeros(n, bool)
            if ev is not None:
                for r in ev[ev.stream_id == s.stream_id].itertuples():
                    y |= (t >= r.t_start) & (t < r.t_end)
            d = pd.DataFrame({**f, **st})
            d["sid"], d["t"], d["y"] = s.stream_id, t, y.astype(int)
            d["_prev"], d["_next"] = prev, nxt
            frames.append((d, amp))
    finally:
        P.FEATS = saved
    return frames


def events_of(df: pd.DataFrame, p: np.ndarray, g) -> pd.DataFrame:
    evs = [e for sid, d in df.assign(p=p).groupby("sid") for e in P.to_events(sid, d.t.values - STEP / 2, d.p.values, *g)]
    return pd.DataFrame(evs, columns=["stream_id", "t_start", "t_end", "score", "t_alarm"]).round(3)


def run_fold(k: int, fs: str) -> dict:
    t0 = time.time()
    tr, pca = P.add_fingerprint(build(C.sources(k, "train"), C.events(k, "train")), None)
    te, _ = P.add_fingerprint(build(C.sources(k, "test"), None), pca)
    cols = P.cols_for(tr, fs)
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(4).split(tr, tr.y, tr.sid):
        m = HistGradientBoostingClassifier(**PARAMS).fit(tr.iloc[a][cols], tr.y.iloc[a])
        oof[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]
    best = None
    for g in GRID:
        sc = S.score(events_of(tr, oof, g), k, part="train")["ranking_score"]["f1_mean"]
        if best is None or sc > best[0]:
            best = (sc, g)
    model = HistGradientBoostingClassifier(**PARAMS).fit(tr[cols], tr.y)
    pred = events_of(te, model.predict_proba(te[cols])[:, 1], best[1])
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT / f"fold_{k}_{fs}.csv", index=False)
    rep = S.summary(S.score(OUT / f"fold_{k}_{fs}.csv", k))
    res = {"fold": k, "train_oof_rank_f1": best[0], "postproc(thr,smooth,min_dur,gap)": best[1], "test": rep,
           "n_train_blocks": len(tr), "n_test_blocks": len(te), "sec": round(time.time() - t0, 1)}
    print(f"[{fs}] fold {k}: train OOF {best[0]:.4f} post={best[1]} | TEST rank F1 {rep['rank_f1']:.4f} "
          f"(F1 {rep['f1@0.1']:.3f}/{rep['f1@0.3']:.3f}/{rep['f1@0.5']:.3f}/{rep['f1@0.7']:.3f}; "
          f"tp/fp/fn@0.3 {rep['tp@0.3']}/{rep['fp@0.3']}/{rep['fn@0.3']}; start med {rep['start_med@0.3']})", flush=True)
    return res


def main() -> None:
    fs = sys.argv[1] if len(sys.argv) > 1 else "full"
    t0 = time.time()
    per_fold = [run_fold(k, fs) for k in range(4)]
    preds = {k: OUT / f"fold_{k}_{fs}.csv" for k in range(4)}
    pooled = S.summary(S.score_pooled(preds))
    ci = S.bootstrap_ci(preds, n_boot=5000, seed=0)
    cc = S.cluster_counts(preds)
    by_subj = cc[cc.theta == "0.3"].groupby("subject")[["tp", "fp", "fn"]].sum()
    out = {"feature_set": fs, "per_fold": per_fold, "pooled": pooled, "bootstrap": ci,
           "fold_rank_f1_mean": float(np.mean([r["test"]["rank_f1"] for r in per_fold])),
           "fold_rank_f1_sd": float(np.std([r["test"]["rank_f1"] for r in per_fold], ddof=1)),
           "pooled_rank_f1_without_Test_P3_3": S.rank_f1_by_subject_mask(preds, ["Test_P3_3"]),
           "by_subject@0.3": by_subj.reset_index().to_dict("records"),
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    (OUT / f"summary_{fs}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[{fs}] POOLED rank F1 {pooled['rank_f1']:.4f} 95% CI [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}] | fold mean "
          f"{out['fold_rank_f1_mean']:.4f} +- {out['fold_rank_f1_sd']:.4f} | wall {out['wall_s']:.0f}s "
          f"peak RSS {out['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()

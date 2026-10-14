"""Score the analysis/prototype_detector.py model on the pseudo-test bank.

Exactly the prototype recipe (same features, HGB params, random_state=0, 4-fold GroupKFold-by-stream OOF on
the 13 train streams, same 80-point post-processing grid selected on train OOF ranking F1). Then:
  * official val score (official CLI on the 5 real val h5)  -> must reproduce reports/prototype/summary.json
  * every pseudo split (seed) scored with scorer.score_csv  (= official evaluate on npz durations)

Usage: python3 eval_prototype.py [physics full ...]
Output: results/prototype_bank.json, results/per_seed.csv, results/per_stream.csv, results/preds/*.csv
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/challenge1/analysis")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import prototype_detector as P  # noqa: E402
import scorer as S  # noqa: E402

PSEUDO = Path("/home/user/data_raw/pseudo")
TRAIN_FEATS = P.FEATS
RES = HERE / "results"
PRED = RES / "preds"
PRED.mkdir(parents=True, exist_ok=True)
STEP = P.STEP


def build_unlabelled(sids: list[str], feats_dir: Path):
    """P.build without frame_labels (pseudo streams have no package composition)."""
    P.FEATS = feats_dir  # block_features reads the module global
    frames = []
    try:
        for sid in sids:
            f, amp = P.block_features(sid)
            s, prev, nxt = P.structure_features(f)
            n = len(f["jump_max"])
            d = pd.DataFrame({**f, **s})
            d["sid"], d["t"], d["y"] = sid, (np.arange(n) + 0.5) * STEP, 0
            d["_prev"], d["_next"] = prev, nxt
            frames.append((d, amp))
    finally:
        P.FEATS = TRAIN_FEATS
    return frames


def events_of(df: pd.DataFrame, p: np.ndarray, g) -> list:
    return [e for sid, d in df.assign(p=p).groupby("sid") for e in P.to_events(sid, d.t.values - STEP / 2, d.p.values, *g)]


def summarise(rep: dict) -> dict:
    bt = rep["by_theta"]
    return {
        "rank_f1": rep["ranking_score"]["f1_mean"],
        "f1_01": bt["0.1"]["f1"], "f1_03": bt["0.3"]["f1"], "f1_05": bt["0.5"]["f1"], "f1_07": bt["0.7"]["f1"],
        "tp03": bt["0.3"]["tp"], "fp03": bt["0.3"]["fp"], "fn03": bt["0.3"]["fn"],
        "start_med03": bt["0.3"]["start_error_median_abs"], "fa_h03": bt["0.3"]["false_alarms_per_hour"],
        "n_ign_pred03": bt["0.3"]["n_ignored_predictions"],
    }


def main() -> None:
    sets = sys.argv[1:] or ["physics", "full"]
    t0 = time.time()
    sids = P.stream_ids()
    tr_ids = [s for s in sids if s.startswith("train")]
    va_ids = [s for s in sids if s.startswith("val")]
    tr, pca = P.add_fingerprint(P.build(tr_ids), None)
    va, _ = P.add_fingerprint(P.build(va_ids), pca)
    tr_dur = S.durations_from_npz(TRAIN_FEATS, tr_ids)
    seeds = sorted(int(p.stem.split("_")[1]) for p in PSEUDO.glob("events_*.csv"))
    print(f"train/val built {time.time() - t0:.0f}s; pseudo seeds {seeds}", flush=True)
    grid = [(thr, sm, md, gp) for thr in (0.3, 0.4, 0.5, 0.6, 0.7) for sm in (1, 3, 5) for md in (0.4, 0.8) for gp in (0.2, 0.6)]
    params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    out = {"seeds": seeds, "sets": {}}
    rows_seed, rows_stream = [], []
    for fs in sets:
        cols = P.cols_for(tr, fs)
        oof = np.zeros(len(tr))
        for a, b in GroupKFold(4).split(tr, tr.y, tr.sid):
            m = HistGradientBoostingClassifier(**params).fit(tr.iloc[a][cols], tr.y.iloc[a])
            oof[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]
        best = None
        for g in grid:
            P.write_sub(events_of(tr, oof, g), RES / "tmp_train.csv")
            sc = S.score_csv(P.PKG / "streams_events.csv", RES / "tmp_train.csv", tr_dur)["ranking_score"]["f1_mean"]
            if best is None or sc > best[0]:
                best = (sc, g)
        model = HistGradientBoostingClassifier(**params).fit(tr[cols], tr.y)
        # official val (CLI on real h5)
        P.write_sub(events_of(va, model.predict_proba(va[cols])[:, 1], best[1]), PRED / f"val_{fs}.csv")
        val = summarise(S.official_cli(P.PKG / "streams_events.csv", PRED / f"val_{fs}.csv", P.PKG / "streams" / "val",
                                       RES / f"val_{fs}_official.json"))
        print(f"[{fs}] train OOF {best[0]:.4f} post={best[1]} | VAL official {val['rank_f1']:.4f}", flush=True)
        # pseudo bank, one seed (= one pseudo val split) at a time to bound RAM
        per_seed = []
        for seed in seeds:
            ids = sorted(p.stem for p in (PSEUDO / "feats").glob(f"p{seed}_*.npz") if ".tmp" not in p.name)
            ps, _ = P.add_fingerprint(build_unlabelled(ids, PSEUDO / "feats"), pca)
            pred_csv = PRED / f"pseudo_{seed}_{fs}.csv"
            P.write_sub(events_of(ps, model.predict_proba(ps[cols])[:, 1], best[1]), pred_csv)
            dur = S.durations_from_npz(PSEUDO / "feats", ids)
            gt = PSEUDO / f"events_{seed}.csv"
            r = summarise(S.score_csv(gt, pred_csv, dur))
            per_seed.append(r)
            rows_seed.append({"set": fs, "seed": seed, "n_streams": len(ids), "hours": sum(dur.values()) / 3600, **r})
            allp = pd.read_csv(pred_csv)
            for sid in ids:  # per-stream scores (same official code restricted to one stream)
                one = RES / "tmp_stream.csv"
                allp[allp.stream_id == sid].to_csv(one, index=False)
                rows_stream.append({"set": fs, "seed": seed, "stream_id": sid,
                                    **summarise(S.score_csv(gt, one, {sid: dur[sid]}))})
            print(f"  [{fs}] seed {seed}: rank F1 {r['rank_f1']:.4f}  "
                  f"(F1 {r['f1_01']:.3f}/{r['f1_03']:.3f}/{r['f1_05']:.3f}) tp/fp/fn {r['tp03']}/{r['fp03']}/{r['fn03']}",
                  flush=True)
            del ps
        rf = np.array([r["rank_f1"] for r in per_seed])
        st = pd.DataFrame([r for r in rows_stream if r["set"] == fs])
        out["sets"][fs] = {
            "train_oof_rank_f1": best[0], "postproc(thr,smooth,min_dur,gap)": best[1], "val_official": val,
            "bank_per_split_rank_f1_mean": float(rf.mean()), "bank_per_split_rank_f1_sd": float(rf.std(ddof=1)),
            "bank_per_split_min_max": [float(rf.min()), float(rf.max())],
            "bank_n_splits": len(rf), "bank_n_streams": int(len(st)),
            "bank_per_stream_rank_f1_mean": float(st.rank_f1.mean()), "bank_per_stream_rank_f1_sd": float(st.rank_f1.std()),
            "bank_mean_f1_by_theta": {k: float(np.mean([r[k] for r in per_seed])) for k in ("f1_01", "f1_03", "f1_05", "f1_07")},
            "bank_mean_start_med03": float(np.mean([r["start_med03"] for r in per_seed])),
            "bank_mean_fa_h03": float(np.mean([r["fa_h03"] for r in per_seed])),
            "val_minus_bank_mean_in_sd": float((val["rank_f1"] - rf.mean()) / rf.std(ddof=1)),
        }
        print(f"[{fs}] BANK per-split rank F1 {rf.mean():.4f} +- {rf.std(ddof=1):.4f} (n={len(rf)}); "
              f"per-stream {st.rank_f1.mean():.4f} +- {st.rank_f1.std():.4f} (n={len(st)})", flush=True)
    if len(sets) > 1:
        a, b = sets[0], sets[-1]
        ds = pd.DataFrame(rows_seed)
        d = ds[ds.set == b].set_index("seed").rank_f1 - ds[ds.set == a].set_index("seed").rank_f1
        out[f"paired_{b}_minus_{a}"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "n_pos": int((d > 0).sum()),
                                        "n": int(len(d)),
                                        "val_official": out["sets"][b]["val_official"]["rank_f1"] - out["sets"][a]["val_official"]["rank_f1"]}
        print("paired", json.dumps(out[f"paired_{b}_minus_{a}"]), flush=True)
    out["wall_s"] = time.time() - t0
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    pd.DataFrame(rows_seed).to_csv(RES / "per_seed.csv", index=False)
    pd.DataFrame(rows_stream).to_csv(RES / "per_stream.csv", index=False)
    (RES / "prototype_bank.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"done {out['wall_s']:.0f}s peak RSS {out['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()

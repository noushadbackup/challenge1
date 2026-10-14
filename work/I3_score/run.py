"""I3 (PLAN v2 W5 / X4): event hit model as score column + nested operating threshold, on the reference (G7N4).

Per (bag, target): reference candidates (G7-gated blend at the I1 G7 operating point) of the train OOF and of the
test part, replay internals from work/I2_boundaries/cache/rep_<t>_{train,test}.pkl (byte-checked there against
I1 G7N4), hit-model features (uwbfall.hitmodel.candidate_features), labels on the refined rows of the training
candidates (official ignore rule: free ignore-zone rows left out).

Variants (all end with the reference time NMS 4 s; the NMS keeps the higher *score* of the variant):
  REF  detector score (must reproduce I1 G7N4 byte-for-byte; checked)
  T0   REF + cut detector score < tau, tau on train OOF final rows (no model -> no nesting needed)
  H    score = hit model (fit on all train candidates)
  HT   H + cut hit score < tau, tau on train OOF rows scored by subject-grouped inner-CV hit models (nested)

    python3 work/I3_score/run.py <targets> [--bags ...]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
from common import BAGS, COLS, I1, REF_CFG, Probs, compositions, oname, ops_g7, target_sources  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall import hitmodel as HM  # noqa: E402
from uwbfall.decode import events_df, to_events  # noqa: E402
from uwbfall.scoring import gt_records, score_inproc, summary  # noqa: E402
from uwbfall.sources import events_for  # noqa: E402

I2C = HERE.parent / "I2_boundaries" / "cache"
KEY = ["stream_id", "t_start", "t_end"]
CFG_S = {"hit_model": "lgbm", "threshold": "nested", "seeds": 3, "inner_folds": 5, "tau_grid": [0.0, 0.6, 0.02],
         "lgbm": {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 20}}


def subjects(cands: pd.DataFrame, comp: pd.DataFrame) -> np.ndarray:
    out = []
    for r in cands.itertuples():
        c = comp[comp.stream_id == r.stream_id]
        ov = np.minimum(c.t_end.values, r.t_end) - np.maximum(c.t_start.values, r.t_start)
        out.append(CL.subject_of_path(c.path.values[int(np.argmax(ov))]))
    return np.array(out)


def nms(df):
    return CL.nms_time(df, REF_CFG["decode"]["nms_gap_s"]).reset_index(drop=True)


def write(bag, name, t, df):
    d = HERE / "preds" / bag / name
    d.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["stream_id", "t_start"], kind="stable") if name != "REF" else df
    df[COLS].to_csv(d / f"{oname(t)}.csv", index=False)
    return d / f"{oname(t)}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("--bags", default=",".join(BAGS))
    a = ap.parse_args()
    b = REF_CFG["boundary"]
    for t in a.targets.split(","):
        t0 = time.time()
        tr, evtr, te, _ = target_sources(t)
        te = sorted(te, key=lambda s: s.sid)
        gt = gt_records(events_for(tr, evtr), ignore_flagged=True)
        dur_tr = {s.sid: s.duration for s in tr}
        dur_te = {s.sid: s.duration for s in te}
        comp_tr, _ = compositions(t)
        Rtr = pickle.load(open(I2C / f"rep_{t}_train.pkl", "rb"))
        Rte = pickle.load(open(I2C / f"rep_{t}_test.pkl", "rb"))
        prep = pickle.load(open(I1 / "cache" / f"prep_{t}.pkl", "rb"))
        peaks = prep["peaks"]
        info = {}
        for bag in a.bags.split(","):
            P = Probs(bag, t)
            op = ops_g7(bag, t)
            assert list(Rtr["ops"][bag]) == list(op)
            # ---- train OOF candidates
            ctr = Rtr["cand"][bag]
            Mtr = ctr.merge(Rtr["D"], on=KEY, how="left")
            assert len(Mtr) == len(ctr) and Mtr[f"{b['crops'][0]}_s"].notna().sum() + Mtr[f"{b['crops'][0]}_s"].isna().sum() == len(Mtr)
            ptr = P.oof()
            parts_tr = {s: {"tcn": P.oof_tcn[P.sid == s], "hgb": P.oof_hgb[P.sid == s], "gate": P.gates[s]} for s in ptr}
            Xtr = HM.candidate_features(Mtr, b, ptr, parts_tr, peaks)
            rows_tr = HM.final_rows(Mtr, b, dur_tr)
            Ytr = HM.hit_labels(rows_tr.stream_id.values, rows_tr.t_start.values, rows_tr.t_end.values, gt)
            Xtr_r = Xtr.iloc[rows_tr.row.values].reset_index(drop=True)
            grp = subjects(ctr.iloc[rows_tr.row.values], comp_tr)
            # ---- test candidates
            cte = events_df(sum((to_events(s.sid, P.test(s.sid), *op) for s in te), []))
            Mte = cte.merge(Rte["D"], on=KEY, how="left")
            assert Mte[f"{b['crops'][0]}_s"].notna().any()
            pte = {s.sid: P.test(s.sid) for s in te}
            parts_te = {s.sid: {"tcn": P.Pt[f"{s.sid}|tcn"], "hgb": P.Pt[f"{s.sid}|hgb"], "gate": P.gates[s.sid]}
                        for s in te}
            Xte = HM.candidate_features(Mte, b, pte, parts_te, peaks)
            rows_te = HM.final_rows(Mte, b, dur_te)
            Xte_r = Xte.iloc[rows_te.row.values].reset_index(drop=True)
            pickle.dump({"Xtr": Xtr_r, "Ytr": Ytr, "grp": grp, "rows_tr": rows_tr, "Xte": Xte_r, "rows_te": rows_te},
                        open(HERE / "cache" / f"feat_{bag}_{t}.pkl", "wb"))
            # REF
            ref = nms(rows_te.drop(columns="row"))
            f = write(bag, "REF", t, ref)
            refb = (I1 / "preds" / bag / "G7N4" / f"{oname(t)}.csv").read_bytes()
            same = f.read_bytes() == refb
            # T0: threshold on the detector score (train OOF final rows)
            tau0, curve0 = HM.choose_threshold(rows_tr.drop(columns="row"), gt, dur_tr, CFG_S, nms)
            write(bag, "T0", t, ref[ref.score >= tau0])
            # H: hit-model score
            m = Ytr.train_ok.values
            hm = HM.HitModel(CFG_S).fit(Xtr_r[m], Ytr[m])
            h_te = hm.predict(Xte_r)
            H = nms(rows_te.drop(columns="row").assign(score=np.round(h_te, 6)))
            write(bag, "H", t, H)
            # HT: nested threshold on inner-OOF hit scores
            h_oof = HM.inner_oof(Xtr_r, Ytr, grp, CFG_S, CFG_S["inner_folds"])
            tau, curve = HM.choose_threshold(rows_tr.drop(columns="row").assign(score=np.round(h_oof, 6)), gt, dur_tr,
                                             CFG_S, nms)
            write(bag, "HT", t, H[H.score >= tau])
            from sklearn.metrics import roc_auc_score
            auc = {str(th): float(roc_auc_score(Ytr[f"hit{th}"][m], h_oof[m])) if Ytr[f"hit{th}"][m].nunique() > 1 else None
                   for th in HM.THETAS}
            auc_det = {str(th): float(roc_auc_score(Ytr[f"hit{th}"][m], rows_tr.score[m])) if Ytr[f"hit{th}"][m].nunique() > 1 else None
                       for th in HM.THETAS}
            tr_ref = summary(score_inproc(nms(rows_tr.drop(columns="row")), gt, dur_tr))
            info[bag] = {"ref_identical_to_I1_G7N4": same, "op": list(op), "n_train_cand": int(len(rows_tr)),
                         "n_train_ok": int(m.sum()), "n_train_ignore_free": int((~m).sum()),
                         "train_hit_rate": {str(th): float(Ytr[f"hit{th}"][m].mean()) for th in HM.THETAS},
                         "tau_T0": tau0, "tau_HT": tau, "curve_T0": curve0, "curve_HT": curve,
                         "inner_auc_hit": auc, "train_auc_detector_score": auc_det, "train_oof_ref": tr_ref,
                         "importance": hm.importance(), "n_test_cand": int(len(rows_te))}
            print(f"{t} {bag}: REF identical={same} tau_T0={tau0} tau_HT={tau} innerAUC={auc} detAUC={auc_det} "
                  f"n_tr={len(rows_tr)} free={int((~m).sum())}", flush=True)
        (HERE / "out" / f"info_{t}.json").write_text(json.dumps(info, indent=1))
        print(f"target {t} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

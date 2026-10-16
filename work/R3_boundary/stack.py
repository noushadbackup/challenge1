"""Stacked boundary regressor (learned replacement of the fixed detector/replay blend).

For every candidate whose replay passed the agreement gate, a robust linear model (HuberRegressor on
standardised features) predicts the GT start / end offset relative to the replay median from
  'det'  : only the detector-minus-replay offsets (s_det - s_rep, e_det - e_rep)  -> a learned wdet + bias
  'full' : + the 6 per-crop offsets to the median (start and end), crop-agreement mask, per-crop spread / low-q,
           number of agreeing crops, replay and detector durations, detector score
Labels: training-OOF candidates (D_train) whose replay bounds have tIoU >= 0.3 with an evaluable GT fall.
Gate-failed candidates keep the detector bounds (as in the base).  Everything is fitted on the target's TRAINING
streams; the training-OOF score is an inner stream-fold CV (bundle meta folds) so it is comparable to wdet's.

    python3 stack.py det full      -> results/stack_<kind>/{fold_k,val}.csv + summary.json
"""
from common import *  # noqa
import json
import srcv
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from uwbfall.boundary import refine, to_preds
from uwbfall.pipeline import load_bundle, _nms
from uwbfall.scoring import gt_records, score_inproc, summary

CLIP_X, CLIP_Y = 3.0, 2.0


def feats(D, b, kind):
    a, e, use = refine(D, b)
    s0, e0 = D.t_start.values.astype(float), D.t_end.values.astype(float)
    cols = [np.clip(s0 - a, -CLIP_X, CLIP_X), np.clip(e0 - e, -CLIP_X, CLIP_X)]
    if kind == "full":
        S = np.vstack([D[f"{c}_s"].values for c in b["crops"]]).astype(float)
        E = np.vstack([D[f"{c}_e"].values for c in b["crops"]]).astype(float)
        ok = np.isfinite(S) & np.isfinite(E) & (E > S) & (tiou(S, E, s0, e0) >= b["gate"])
        cols += list(np.where(ok, np.clip(S - a, -CLIP_X, CLIP_X), 0.0))
        cols += list(np.where(ok, np.clip(E - e, -CLIP_X, CLIP_X), 0.0))
        cols += list(ok.astype(float))
        cols += [np.nan_to_num(D[f"{c}_spread"].values.astype(float)).clip(0, 5) for c in b["crops"]]
        cols += [D[f"{c}_lowq"].astype(str).str.lower().eq("true").values.astype(float) for c in b["crops"]]
        cols += [ok.sum(0).astype(float), (e - a), (e0 - s0).clip(0, 20), D.score.values.astype(float)]
    return np.vstack(cols).T, a, e, use


def labels(D, a, e, ev):
    ev = unflagged(ev)
    ys = np.full(len(D), np.nan); ye = np.full(len(D), np.nan)
    for i in range(len(D)):
        g = ev[ev.stream_id == D.stream_id.iat[i]]
        if not len(g):
            continue
        t = tiou(a[i], e[i], g.t_start.values.astype(float), g.t_end.values.astype(float))
        j = int(np.argmax(t))
        if t[j] >= 0.3:
            ys[i] = np.clip(float(g.t_start.iat[j]) - a[i], -CLIP_Y, CLIP_Y)
            ye[i] = np.clip(float(g.t_end.iat[j]) - e[i], -CLIP_Y, CLIP_Y)
    return ys, ye


def fit(X, ys, ye, m):
    mk = lambda: make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=2000))
    return mk().fit(X[m], ys[m]), mk().fit(X[m], ye[m])


def apply_bounds(X, a, e, use, models, dmin):
    ms, me = models
    ns, ne = a.copy(), e.copy()
    if use.any():
        ns[use] = a[use] + np.clip(ms.predict(X[use]), -CLIP_Y, CLIP_Y)
        ne[use] = e[use] + np.clip(me.predict(X[use]), -CLIP_Y, CLIP_Y)
    return ns, np.maximum(ne, ns + dmin)


def run(kind):
    out = HERE / "results" / f"stack_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    preds, info = {}, {}
    for t in TARGETS:
        B = load_bundle(bundle_dir(t)); cfg = B["cfg"]; b = cfg["boundary"]
        Dt = pd.read_csv(DUMP / tname(t) / "D_train.csv")
        X, a, e, use = feats(Dt, b, kind)
        ys, ye = labels(Dt, a, e, train_events(t))
        lab = np.isfinite(ys) & use
        # inner stream-fold CV on the training streams (training-OOF estimate)
        ns, ne = a.copy(), np.maximum(e, a + b["dmin"])
        for k, held in B["meta"]["folds"].items():
            h = Dt.stream_id.isin(held).values
            mods = fit(X, ys, ye, lab & ~h)
            s_, e_ = apply_bounds(X, a, e, use & h, mods, b["dmin"])
            ns[h], ne[h] = s_[h], e_[h]
        dur = {s.sid: s.duration for s in train_sources(t)}
        gt = gt_records(train_events(t), ignore_flagged=True)
        tr = summary(score_inproc(_nms(to_preds(Dt, ns, ne, dur), cfg, {}), gt, dur))
        base_tr = summary(score_inproc(_nms(to_preds(Dt, a, np.maximum(e, a + b["dmin"]), dur), cfg, {}), gt, dur))
        # final: fit on all training candidates, apply to the test candidates
        mods = fit(X, ys, ye, lab)
        D = pd.read_csv(DUMP / tname(t) / "D_test.csv")
        Xs, as_, es, us = feats(D, b, kind)
        s_, e_ = apply_bounds(Xs, as_, es, us, mods, b["dmin"])
        r = _nms(to_preds(D, s_, e_, {s.sid: s.duration for s in test_sources(t)}), cfg, {})
        f = out / ("val.csv" if t == "val" else f"fold_{t}.csv")
        r.to_csv(f, index=False)
        preds[t] = f
        info[tname(t)] = {"n_train_labels": int(lab.sum()), "train_oof_inner_cv": tr, "train_oof_base": base_tr,
                          "coef_start_det": float(mods[0][-1].coef_[0] / mods[0][0].scale_[0]),
                          "coef_end_det": float(mods[1][-1].coef_[1] / mods[1][0].scale_[1])}
        print(tname(t), kind, "train-OOF", base_tr["rank_f1"], "->", tr["rank_f1"], tr["f1@0.1/0.3/0.5/0.7"],
              "n_lab", int(lab.sum()), flush=True)
    folds = {k: preds[k] for k in range(4)}
    res = {"variant": f"stack_{kind}", "train": info, "pooled": summary(srcv.score_pooled(folds)),
           "per_fold": {k: summary(srcv.score(folds[k], k)) for k in range(4)}, "val": summary(srcv.score_val(preds["val"])),
           "promotion_vs_final_norep": srcv.promote({k: base_csv(k) for k in range(4)}, folds,
                                                    val_A=base_csv("val"), val_B=preds["val"])}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("pooled", "val")}, default=float))
    pr = res["promotion_vs_final_norep"]
    print(json.dumps({k: pr[k] for k in ("delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved", "val_B", "promote")}
                     | {"per_fold": [round(x["delta"], 4) for x in pr["per_fold"]]}, default=float), flush=True)


if __name__ == "__main__":
    for kd in sys.argv[1:]:
        run(kd)

"""Complementarity of the TCN and the HGB prototype + probability-averaging ensemble.

  python3 compare.py <tcn_tag> [w_tcn ...]
Block level (train OOF): AUC/AP of each model, Pearson r of probabilities and of residuals (y - p).
Event level (val, official evaluator CSVs): per evaluable GT fall, best tIoU of each model -> overlap of misses.
Ensemble: p = w*tcn + (1-w)*hgb; post-processing re-tuned on the averaged train OOF; val scored once per w.
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from lib import GT, OUT, events_from_probs, load, official, summary, tune  # noqa: E402

tag = sys.argv[1]
weights = [float(w) for w in sys.argv[2:]] or [0.5]
D = load()
y, yv = D["train_y"], D["val_y"]
H, T = np.load(OUT / "probs_hgb.npz"), np.load(OUT / f"probs_{tag}.npz")
res = {"block_oof": {}, "block_val": {}}
for nm, P in (("hgb", H), ("tcn", T)):
    res["block_oof"][nm] = {"auc": round(roc_auc_score(y, P["oof"]), 4), "ap": round(average_precision_score(y, P["oof"]), 4)}
    res["block_val"][nm] = {"auc": round(roc_auc_score(yv, P["val"]), 4), "ap": round(average_precision_score(yv, P["val"]), 4)}
for split, key, yy in (("oof", "oof", y), ("val", "val", yv)):
    a, b = H[key], T[key]
    res[f"corr_{split}"] = {"prob_pearson": round(float(np.corrcoef(a, b)[0, 1]), 3),
                            "residual_pearson": round(float(np.corrcoef(yy - a, yy - b)[0, 1]), 3),
                            "residual_pearson_on_falls": round(float(np.corrcoef((yy - a)[yy == 1], (yy - b)[yy == 1])[0, 1]), 3)}


def best_tiou(pred_csv):
    gt = pd.read_csv(GT)
    gt = gt[gt.stream_id.str.startswith("val") & (gt.quality != "low") & ~gt.needs_review.astype(str).str.lower().eq("true")]
    pr = pd.read_csv(pred_csv)
    out = []
    for r in gt.itertuples():
        q = pr[pr.stream_id == r.stream_id]
        inter = np.clip(np.minimum(q.t_end, r.t_end) - np.maximum(q.t_start, r.t_start), 0, None)
        uni = np.maximum(q.t_end, r.t_end) - np.minimum(q.t_start, r.t_start)
        out.append(float((inter / uni).max()) if len(q) else 0.0)
    return np.array(out)


bh, bt = best_tiou(OUT / "val_hgb.csv"), best_tiou(OUT / f"val_{tag}.csv")
res["val_events"] = {}
for th in (0.3, 0.5):
    mh, mt = bh < th, bt < th
    res["val_events"][f"miss@{th}"] = {"hgb": int(mh.sum()), "tcn": int(mt.sum()), "both": int((mh & mt).sum()),
                                       "only_hgb": int((mh & ~mt).sum()), "only_tcn": int((~mh & mt).sum())}
res["val_events"]["n_evaluable"] = len(bh)
res["val_events"]["tiou_pearson"] = round(float(np.corrcoef(bh, bt)[0, 1]), 3)
res["ensemble"] = {}
# weight chosen on train OOF only; w=0.5 is the pre-registered default and is always scored too
oof_scores = {}
for w in weights:
    po = w * T["oof"] + (1 - w) * H["oof"]
    oof_scores[w] = tune(D["train_sid"], D["train_t"], po)
    print("oof", w, oof_scores[w], flush=True)
res["ensemble_oof"] = {f"w_tcn={w:g}": {"train_oof_rank_f1": round(v[0], 4), "postproc": v[1]} for w, v in oof_scores.items()}
w_best = max(oof_scores, key=lambda w: oof_scores[w][0])
res["ensemble_w_chosen_on_oof"] = w_best
for w in sorted({0.5, w_best}):
    sc, g = oof_scores[w] if w in oof_scores else tune(D["train_sid"], D["train_t"], w * T["oof"] + (1 - w) * H["oof"])
    pv = w * T["val"] + (1 - w) * H["val"]
    r = official(events_from_probs(D["val_sid"], D["val_t"], pv, g), f"val_ens_{tag}_w{w:g}")
    res["ensemble"][f"w_tcn={w:g}"] = {"train_oof_rank_f1": round(sc, 4), "postproc": g, "val": summary(r)}
    print(w, res["ensemble"][f"w_tcn={w:g}"], flush=True)
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / f"compare_{tag}.json", "w"), indent=1)

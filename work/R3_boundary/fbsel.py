"""Idea (i) done properly: on top of the wdet_cv blend, choose the fallback for gate-failed candidates
(detector bounds vs dense boundary model b3 'argmax' / 'exp' bounds) per target on its TRAINING OOF
(ties -> detector), then apply to the target's test candidates.

    python3 fbsel.py            -> results/wdet_fb/{fold_k,val}.csv + summary.json
"""
from common import *  # noqa
import json
import srcv
from apply import choose
from select_b import cand_model, final_rows
from uwbfall.pipeline import load_bundle
from uwbfall.scoring import gt_records, score_inproc, summary

out = HERE / "results" / "wdet_fb"
out.mkdir(parents=True, exist_ok=True)
chosen, preds = {}, {}
for t in TARGETS:
    B = load_bundle(bundle_dir(t)); cfg = B["cfg"]; b = cfg["boundary"]
    v0, _ = choose("wdet_cv", t)
    Dt = pd.read_csv(DUMP / tname(t) / "D_train.csv")
    Mt = cand_model(Dt, dict(np.load(BIG / "logits" / "b3" / tname(t) / "train.npz")))
    dur = {s.sid: s.duration for s in train_sources(t)}
    gt = gt_records(train_events(t), ignore_flagged=True)
    tr = {}
    for fb in ("det", "model:argmax", "model:exp"):
        v = {**v0, "fb": fb.split(":")[0], "how": fb.split(":")[-1]}
        tr[fb] = summary(score_inproc(final_rows(Dt, b, Mt, v, dur, cfg), gt, dur))["rank_f1"]
    best = max(tr, key=lambda k: (tr[k], k == "det"))
    v = {**v0, "fb": best.split(":")[0], "how": best.split(":")[-1]}
    chosen[tname(t)] = {"params": v, "train_oof_rank": tr}
    D = pd.read_csv(DUMP / tname(t) / "D_test.csv")
    M = cand_model(D, dict(np.load(BIG / "logits" / "b3" / tname(t) / "test.npz")))
    r = final_rows(D, b, M, v, {s.sid: s.duration for s in test_sources(t)}, cfg)
    f = out / ("val.csv" if t == "val" else f"fold_{t}.csv")
    r.to_csv(f, index=False)
    preds[t] = f
    print(tname(t), chosen[tname(t)], flush=True)
folds = {k: preds[k] for k in range(4)}
res = {"variant": "wdet_fb", "chosen_on_training_oof": chosen, "pooled": summary(srcv.score_pooled(folds)),
       "per_fold": {k: summary(srcv.score(folds[k], k)) for k in range(4)}, "val": summary(srcv.score_val(preds["val"])),
       "promotion_vs_final_norep": srcv.promote({k: base_csv(k) for k in range(4)}, folds,
                                                val_A=base_csv("val"), val_B=preds["val"])}
(out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res[k] for k in ("pooled", "val")}, default=float))
print(json.dumps(res["promotion_vs_final_norep"], default=float)[:1200])

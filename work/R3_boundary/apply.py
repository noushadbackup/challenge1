"""Choose each variant's parameters on the target's TRAINING-OOF grid (grid2_<tag>.json), apply them to the
target's test candidates (D_test), write results/<variant>/{fold_k,val}.csv, score SR-CV pooled + val and run
the promotion rule against the base final_norep.

    python3 apply.py <variant> [<variant> ...]
variants:
  wdet_cv     per target argmax_(ws,we) training-OOF rank F1 (ties -> smaller ws+we, then smaller ws), lam = 0
  wdet_sym    per target argmax_w with ws = we = w
  model_cv    per target argmax_(ws,we,lam) with the dense boundary model b3 (ties -> smaller lam, ws+we)
  model_only  per target argmax_lam with ws = we = 0 (replay blended with the boundary model only)
  fb_model    gate-failed candidates take the b3 model bounds (no blending)
  fixed:<ws>:<we>  fixed weights (sensitivity only: chosen after looking at all targets' training OOF)
"""
from common import *  # noqa
import json
import srcv
from select_b import cand_model, final_rows
from uwbfall.pipeline import load_bundle
from uwbfall.scoring import summary


def choose(variant, t):
    tag = "b3" if variant.startswith("model") else "none"
    R = pd.DataFrame(json.load(open(HERE / f"grid2_{tag}{'_' + BAG if BAG else ''}.json"))[tname(t)])
    if variant == "wdet_sym":
        R = R[(R.ws == R.we) & (R.lam == 0)]
    elif variant == "wdet_cv":
        R = R[R.lam == 0]
    elif variant == "model_only":
        R = R[(R.ws == 0) & (R.we == 0)]
    R = R.assign(tot=R.ws + R.we).sort_values(["rank", "lam", "tot", "ws"], ascending=[False, True, True, True])
    r = R.iloc[0]
    return {"wdet_s": float(r.ws), "wdet_e": float(r.we), "lam": float(r.lam)}, float(r["rank"])


def run(variant):
    out = HERE / "results" / (variant.replace(":", "_") + (f"_{BAG}" if BAG else ""))
    out.mkdir(parents=True, exist_ok=True)
    chosen, preds = {}, {}
    for t in TARGETS:
        B = load_bundle(bundle_dir(t)); cfg = B["cfg"]; b = cfg["boundary"]
        D = pd.read_csv(DUMP / tname(t) / "D_test.csv")
        M = None
        if variant.startswith("fixed:"):
            _, ws, we = variant.split(":")
            v, tr = {"wdet_s": float(ws), "wdet_e": float(we)}, None
        elif variant == "fb_model":
            v, tr = {"fb": "model"}, None
        else:
            v, tr = choose(variant, t)
        if v.get("lam", 0) > 0 or v.get("fb") == "model":
            M = cand_model(D, dict(np.load(BIG / "logits" / "b3" / tname(t) / "test.npz")))
        chosen[tname(t)] = {"params": v, "train_oof_rank": tr}
        dur = {s.sid: s.duration for s in test_sources(t)}
        r = final_rows(D, b, M, v, dur, cfg)
        f = out / ("val.csv" if t == "val" else f"fold_{t}.csv")
        r.to_csv(f, index=False)
        preds[t] = f
    folds = {k: preds[k] for k in range(4)}
    bcsv = (lambda t: DUMP / tname(t) / "base_test.csv") if BAG else base_csv
    base = {k: bcsv(k) for k in range(4)}
    pooled = srcv.score_pooled(folds)
    pr = srcv.promote(base, folds, val_A=bcsv("val"), val_B=preds["val"])
    vb = srcv.score_val(preds["val"])
    res = {"variant": variant, "chosen_on_training_oof": chosen, "pooled": summary(pooled),
           "per_fold": {k: summary(srcv.score(folds[k], k)) for k in range(4)},
           "val": summary(vb), "promotion_vs_final_norep": pr}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"variant": variant, "chosen": chosen, "pooled": res["pooled"], "val": res["val"]}, default=float))
    print(json.dumps(pr, default=float)[:1500], flush=True)
    return res


if __name__ == "__main__":
    for v in sys.argv[1:]:
        run(v)

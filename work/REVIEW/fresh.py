"""Fresh SR-CV regeneration (reviewer, pre-registered): same 4 fold subject splits, NEW test composition seeds
9100+k (official prepare_split + finish_plan, split 'test' settings), so none of these 20 streams was seen by any
round-2 decision. Predict with the existing bag-0 fold bundles of C3 (srcv/runs/uwb_default) and FINAL
(srcv/runs/final_b0); score with the official evaluator; srcv.promote (subject-cluster bootstrap) C3 -> FINAL."""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np
import srcv.core as C
OUT = Path("/home/user/data_raw/review_fresh")
C.DATA = OUT
SEED0 = 9100
_orig_seed = C._seed
C._seed = lambda fold, part: SEED0 + fold if part == "test" else _orig_seed(fold, part)
from srcv import bridge, scoring as S
import uwbfall
from uwbfall.pipeline import predict, load_bundle
stage = sys.argv[1]
systems = {"C3": "/home/user/data_raw/srcv/runs/uwb_default/fold_{k}", "FINAL": "/home/user/data_raw/srcv/runs/final_b0/fold_{k}"}
if stage == "gen":
    for k in range(4):
        m = C.generate(k, "test", OUT / f"fold_{k}")
        print(k, [(s["stream_id"], s["n_frames"], s["n_evaluable"]) for s in m["streams"]], flush=True)
elif stage == "pred":
    for k in [int(x) for x in sys.argv[2].split(",")]:
        srcs = bridge.pipeline_sources(k, "test")
        for name, b in systems.items():
            o = OUT / "preds" / name / f"fold_{k}.csv"
            if o.exists():
                continue
            o.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            p = predict(b.format(k=k), srcs)
            p.to_csv(o, index=False)
            print(k, name, len(p), round(time.time() - t0), "s", flush=True)
elif stage == "score":
    res = {}
    for name in systems:
        d = {k: str(OUT / "preds" / name / f"fold_{k}.csv") for k in range(4)}
        r = S.score_pooled(d); bt = r["by_theta"]
        res[name] = {"pooled": r["ranking_score"]["f1_mean"], "f1": [bt[t]["f1"] for t in ("0.1", "0.3", "0.5", "0.7")],
                     "tp_fp_fn@0.3": [bt["0.3"][x] for x in ("tp", "fp", "fn")], "fa_h": bt["0.3"]["false_alarms_per_hour"],
                     "start_med": bt["0.3"]["start_error_median_abs"],
                     "folds": [S.score(d[k], k)["ranking_score"]["f1_mean"] for k in range(4)],
                     "ci": S.bootstrap_ci(d)["ci95"]}
        print(name, json.dumps(res[name]))
    A = {k: str(OUT / "preds" / "C3" / f"fold_{k}.csv") for k in range(4)}
    B = {k: str(OUT / "preds" / "FINAL" / f"fold_{k}.csv") for k in range(4)}
    pr = S.promote(A, B)
    print("promote C3->FINAL", json.dumps({k: pr[k] for k in ("delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved")}),
          [round(x["delta"], 4) for x in pr["per_fold"]])
    cc = S.cluster_counts(B); t = cc[cc.theta == "0.3"].groupby("subject")[["tp", "fp", "fn"]].sum()
    print(t[(t.fp > 0) | (t.fn > 0)].to_string())
    res["promote"] = pr
    json.dump(res, open(OUT / "fresh_results.json", "w"), indent=1, default=str)

"""Fresh streams, decomposition: C3G3 (uwb_default bundle, boundary gate override 0.3, as `predict --boundary`)
and N4 (= C3G3 + 4 s time NMS, I1 definition). Then promote along the chain C3 -> C3G3 -> N4 -> FINAL."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import pandas as pd
import srcv.core as C
OUT = Path("/home/user/data_raw/review_fresh"); C.DATA = OUT
_o = C._seed; C._seed = lambda f, p: 9100 + f if p == "test" else _o(f, p)
from srcv import bridge, scoring as S
from uwbfall.pipeline import predict, load_bundle
from uwbfall import clip as CL
if sys.argv[1] == "pred":
    for k in range(4):
        o = OUT / "preds" / "C3G3" / f"fold_{k}.csv"
        if not o.exists():
            B = load_bundle(f"/home/user/data_raw/srcv/runs/uwb_default/fold_{k}")
            B["cfg"] = {**B["cfg"], "boundary": {**B["cfg"]["boundary"], "gate": 0.3}}
            t0 = time.time(); p = predict(B, bridge.pipeline_sources(k, "test"))
            o.parent.mkdir(parents=True, exist_ok=True); p.to_csv(o, index=False); print(k, "C3G3", len(p), round(time.time() - t0), flush=True)
        p = pd.read_csv(o)
        n = pd.concat([CL.nms_time(g, 4.0) for _, g in p.groupby("stream_id", sort=True)]).sort_index()
        (OUT / "preds" / "N4").mkdir(parents=True, exist_ok=True); n.to_csv(OUT / "preds" / "N4" / f"fold_{k}.csv", index=False)
else:
    D = lambda nm: {k: str(OUT / "preds" / nm / f"fold_{k}.csv") for k in range(4)}
    res = {}
    for nm in ("C3", "C3G3", "N4", "FINAL"):
        r = S.score_pooled(D(nm)); bt = r["by_theta"]["0.3"]
        res[nm] = {"pooled": r["ranking_score"]["f1_mean"], "tp_fp_fn": [bt["tp"], bt["fp"], bt["fn"]]}
        print(nm, round(res[nm]["pooled"], 4), res[nm]["tp_fp_fn"])
    for a, b in (("C3", "C3G3"), ("C3G3", "N4"), ("N4", "FINAL"), ("C3G3", "FINAL"), ("C3", "FINAL")):
        pr = S.promote(D(a), D(b))
        res[f"{a}->{b}"] = {k: pr[k] for k in ("delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved")}
        res[f"{a}->{b}"]["per_fold"] = [round(x["delta"], 4) for x in pr["per_fold"]]
        print(a, "->", b, json.dumps(res[f"{a}->{b}"]))
    json.dump(res, open(OUT / "fresh_chain.json", "w"), indent=1)

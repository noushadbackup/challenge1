"""Training-OOF grid: separate start/end detector weights (+ optional model weight lam).  python3 grid2.py <tag|none> [targets]"""
from common import *  # noqa
import json
import os
from select_b import cand_model, final_rows
from uwbfall.pipeline import load_bundle
from uwbfall.scoring import gt_records, score_inproc, summary

tag = sys.argv[1]
tg = [("val" if a == "val" else int(a)) for a in sys.argv[2:]] or TARGETS
W = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LAM = [0.0] if tag == "none" else [0.0, 0.1, 0.2, 0.3]
TAU = [float(x) for x in os.environ.get("TAU", "9").split(",")]
res = {}
for t in tg:
    B = load_bundle(bundle_dir(t)); cfg = B["cfg"]; b = cfg["boundary"]
    D = pd.read_csv(DUMP / tname(t) / "D_train.csv")
    M = None if tag == "none" else cand_model(D, dict(np.load(BIG / "logits" / tag / tname(t) / "train.npz")))
    dur = {s.sid: s.duration for s in train_sources(t)}
    gt = gt_records(train_events(t), ignore_flagged=True)
    rows = []
    for ws in W:
        for we in W:
            for lam in LAM:
              for tau in TAU:
                if ws + lam > 0.8 or we + lam > 0.8 or (tau < 9 and ws + we == 0):
                    continue
                r = final_rows(D, b, M, {"wdet_s": ws, "wdet_e": we, "lam": lam, "tau": tau, "onesided": bool(os.environ.get("ONESIDED"))}, dur, cfg)
                sm = summary(score_inproc(r, gt, dur))
                rows.append(dict(ws=ws, we=we, lam=lam, tau=tau, rank=sm["rank_f1"], f=sm["f1@0.1/0.3/0.5/0.7"],
                                 se=sm.get("start_err_med@0.3")))
    R = pd.DataFrame(rows)
    res[tname(t)] = R.to_dict("records")
    print(f"== {tname(t)}  base {R[(R.ws==0)&(R.we==0)&(R.lam==0)]['rank'].iloc[0]}")
    print(R.sort_values("rank", ascending=False).head(8).to_string(index=False), flush=True)
(HERE / f"grid2_{tag}{os.environ.get('GSUF', '')}{'_' + BAG if BAG else ''}.json").write_text(json.dumps(res))

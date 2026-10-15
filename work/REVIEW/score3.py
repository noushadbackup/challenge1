import sys, json
from pathlib import Path
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import srcv.core as C
from srcv import scoring as S
R = "/home/user/data_raw/review_fresh"
out = {}
def run(tag, A, B):
    r = S.promote(A, B)
    out[tag] = {k: r[k] for k in ("pooled_A", "pooled_B", "delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved")}
    out[tag]["per_fold"] = [round(x["delta"], 4) for x in r["per_fold"]]
    print(tag, json.dumps(out[tag]))
run("orig FINAL(gate0.3)->FG1(gate0.1) bag0", {k: f"/home/user/challenge1/work/FINAL/preds/uwb_default/FINAL/fold_{k}.csv" for k in range(4)},
    {k: f"{R}/preds_FG1/orig/fold_{k}.csv" for k in range(4)})
C.DATA = Path(R); C._comp.cache_clear()
run("fresh FINAL(gate0.3)->FG1(gate0.1) bag0", {k: f"{R}/preds/FINAL/fold_{k}.csv" for k in range(4)}, {k: f"{R}/preds_FG1/fresh/fold_{k}.csv" for k in range(4)})
json.dump(out, open("fg1_results.json", "w"), indent=1)

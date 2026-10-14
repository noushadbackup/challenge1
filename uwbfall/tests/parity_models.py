"""Compare a uwbfall bundle trained on the 13 train streams with the E4 / INTEGRATION reference artefacts.
Usage: PYTHONPATH=/home/user/challenge1 python3 -m uwbfall.tests.parity_models <model_dir> [<val_pred.csv> <val_candidates.csv>]
"""
import json
import sys

import numpy as np
import pandas as pd

W = "/home/user/challenge1/work"
md = sys.argv[1]
o = np.load(f"{md}/oof.npz")
H = np.load(f"{W}/E4_deep_model/out/probs_hgb.npz")
T = np.load(f"{W}/E4_deep_model/out/probs_tcn_all.npz")
res = {}
for nm, mine, ref in (("oof_hgb", o["oof_hgb"], H["oof"]), ("oof_tcn", o["oof_tcn"], T["oof"])):
    if len(mine) == 0:
        continue
    res[nm] = {"bit_equal": bool(np.array_equal(mine, ref)), "max_abs_diff": float(np.max(np.abs(mine - ref))),
               "pearson": float(np.corrcoef(mine, ref)[0, 1])}
meta = json.load(open(f"{md}/meta.json"))
res["postproc"] = {"mine": meta["postproc"], "E4_w0.75": [0.5, 5, 0.8, 0.2]}
res["oof_rank_f1_tuned"] = {"mine": meta["oof_rank_f1_tuned"], "E4_w0.75": 0.9428}


def cmp_csv(a, b):
    A, B = pd.read_csv(a), pd.read_csv(b)
    same = len(A) == len(B) and np.allclose(A[["t_start", "t_end", "score"]].values, B[["t_start", "t_end", "score"]].values, atol=1e-9) \
        and (A.stream_id.values == B.stream_id.values).all()
    return {"rows": [len(A), len(B)], "identical": bool(same)}


if len(sys.argv) > 3:
    res["val_final_vs_INTEGRATION"] = cmp_csv(sys.argv[2], f"{W}/INTEGRATION/out/val_tcn75_proto25_replayB.csv")
    res["val_candidates_vs_E4"] = cmp_csv(sys.argv[3], f"{W}/E4_deep_model/out/val_ens_tcn_all_w0.75.csv")
print(json.dumps(res, indent=1))

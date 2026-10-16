"""Score variants against the base (final_norep bag 0) with the promotion rule.

usage: python score.py <variant> [<variant> ...]   -> results/<variant>/promote.json + one summary line each
"""
import json
import os
import sys

import srcv
from srcv import scoring as SC

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/home/user/challenge1/work/STRICT/results/norep"


def main():
    A = {k: f"{BASE}/fold_{k}.csv" for k in range(4)}
    for v in sys.argv[1:]:
        d = f"{HERE}/results/{v}"
        B = {k: f"{d}/fold_{k}.csv" for k in range(4)}
        pr = srcv.promote(A, B, val_A=f"{BASE}/val.csv", val_B=f"{d}/val.csv")
        sb = SC.summary(srcv.score_pooled(B))
        sv = SC.summary(srcv.score_val(f"{d}/val.csv"))
        pr.update({"pooled_B_summary": sb, "val_B_summary": sv})
        json.dump(pr, open(f"{d}/promote.json", "w"), indent=1)
        print(f"{v}: pooled {pr['pooled_B']:.4f} (base {pr['pooled_A']:.4f}) delta {pr['delta']:+.4f} "
              f"CI [{pr['delta_ci95'][0]:+.4f},{pr['delta_ci95'][1]:+.4f}] P {pr['p_delta_gt_0']:.3f} "
              f"folds {pr['n_folds_improved']}/4 {[round(r['delta'], 4) for r in pr['per_fold']]} "
              f"val {pr['val_B']:.4f} (base {pr['val_A']:.4f}) TP/FP/FN@.3 {sb['tp@0.3']}/{sb['fp@0.3']}/{sb['fn@0.3']} "
              f"val {sv['tp@0.3']}/{sv['fp@0.3']}/{sv['fn@0.3']} promote={pr['promote']}", flush=True)


if __name__ == "__main__":
    main()

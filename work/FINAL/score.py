"""FINAL system table + promotion-rule comparison against the round-1 integration system C3.

    python3 work/FINAL/score.py link      # build preds/<bag>/<system> links (FINAL bag 0 = this clean run)
    python3 work/FINAL/score.py check     # clean-run CSVs vs I1's G7N4 bag-0 CSVs (byte compare)
    python3 work/FINAL/score.py table     # -> out/table.json (R2_EVAL score_all: official evaluator, val via CLI)
    python3 work/FINAL/score.py promote   # C3 -> FINAL (3 bags) and bag-0-only, -> out/promote_*.json

Systems:
  C3     = round-1 INTEGRATION system (0.75 TCN + 0.25 HGB + replay B, gate 0.1), work/R2_EVAL/preds/<bag>/C3
  FINAL  = uwbfall/configs/final.json (= reference G7N4). Bag uwb_default (E4 seeds) = the clean end-to-end run
           `srcv.run_pipeline final_b0 --config final.json` (srcv/results/final_b0); bags tcn_bag1/tcn_bag2 =
           I1's G7N4 predictions of the same config from the seed-offset 10000/20000 TCN bags (retune path, parity
           checked by I1: tests/parity_clip_retune).
"""
import filecmp
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "work" / "R2_EVAL"))
import score_all as SA  # noqa: E402

SA.PRED = HERE / "preds"
SA.OUT = HERE / "out"
BAGS = ["uwb_default", "tcn_bag1", "tcn_bag2"]
CLEAN = REPO / "srcv" / "results" / "final_b0"
FILES = [f"fold_{k}.csv" for k in range(4)] + ["val.csv"]


def link():
    for b in BAGS:
        for sysname, src in [("C3", REPO / "work/R2_EVAL/preds" / b / "C3"),
                             ("FINAL", CLEAN if b == "uwb_default" else REPO / "work/I1_clip/preds" / b / "G7N4")]:
            d = SA.PRED / b / sysname
            d.mkdir(parents=True, exist_ok=True)
            for f in FILES:
                p = d / f
                if p.is_symlink() or p.exists():
                    p.unlink()
                os.symlink(os.path.relpath(src / f, d), p)
    print("linked", SA.PRED)


def check():
    ok = True
    for f in FILES:
        a, b = CLEAN / f, REPO / "work/I1_clip/preds/uwb_default/G7N4" / f
        same = filecmp.cmp(a, b, shallow=False)
        ok &= same
        print(f"{f}: clean run {'==' if same else '!='} I1 G7N4 (bag uwb_default)")
    print("CLEAN RUN REPRODUCES I1 G7N4:", ok)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "link":
        link()
    elif cmd == "check":
        check()
    elif cmd == "table":
        sys.argv = ["x", "table", "C3,FINAL", ",".join(BAGS)]
        SA.main()
    elif cmd == "promote":
        sys.argv = ["x", "promote", "C3", "FINAL", ",".join(BAGS)]
        SA.main()
        os.replace(SA.OUT / "promote_C3_vs_FINAL.json", SA.OUT / "promote_C3_vs_FINAL_3bags.json")
        sys.argv = ["x", "promote", "C3", "FINAL", "uwb_default"]
        SA.main()
        os.replace(SA.OUT / "promote_C3_vs_FINAL.json", SA.OUT / "promote_C3_vs_FINAL_bag0.json")

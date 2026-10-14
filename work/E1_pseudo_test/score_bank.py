"""Score one prediction CSV covering pseudo-bank streams (ids p<seed>_<XX>) split by split.

Each seed is one full pseudo-val split (5 streams, 119 falls, 67 evaluable). The score per split uses
the official abc2027.evaluate code (scorer.score_csv, durations from the npz files).

Usage: python3 score_bank.py preds.csv [--out results.json]
       (rows for streams of seeds not present in the CSV are scored as "no predictions")
Optional paired comparison: python3 score_bank.py candidate.csv --ref reference.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scorer as S  # noqa: E402

PSEUDO = Path("/home/user/data_raw/pseudo")


def per_split(pred_csv: Path) -> pd.DataFrame:
    allp = pd.read_csv(pred_csv)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for ev in sorted(PSEUDO.glob("events_*.csv")):
            seed = int(ev.stem.split("_")[1])
            ids = sorted(p.stem for p in (PSEUDO / "feats").glob(f"p{seed}_*.npz") if ".tmp" not in p.name)
            one = Path(tmp) / f"{seed}.csv"
            allp[allp.stream_id.isin(ids)].to_csv(one, index=False)
            rep = S.score_csv(ev, one, S.durations_from_npz(PSEUDO / "feats", ids))
            bt = rep["by_theta"]
            rows.append({"seed": seed, "rank_f1": rep["ranking_score"]["f1_mean"],
                         "f1_01": bt["0.1"]["f1"], "f1_03": bt["0.3"]["f1"], "f1_05": bt["0.5"]["f1"],
                         "start_med03": bt["0.3"]["start_error_median_abs"], "fa_h03": bt["0.3"]["false_alarms_per_hour"]})
    return pd.DataFrame(rows).set_index("seed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = per_split(a.pred)
    res = {"n_splits": len(d), "rank_f1_mean": d.rank_f1.mean(), "rank_f1_sd": d.rank_f1.std(ddof=1),
           "per_split": d.round(4).reset_index().to_dict("records")}
    print(d.round(4).to_string())
    print(f"rank F1 {res['rank_f1_mean']:.4f} +- {res['rank_f1_sd']:.4f} (n={len(d)} splits)")
    if a.ref:
        diff = d.rank_f1 - per_split(a.ref).rank_f1
        res["paired_minus_ref"] = {"mean": diff.mean(), "sd": diff.std(ddof=1), "n_pos": int((diff > 0).sum())}
        print(f"paired delta vs ref: {diff.mean():+.4f} +- {diff.std(ddof=1):.4f}, positive in {(diff > 0).sum()}/{len(diff)}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

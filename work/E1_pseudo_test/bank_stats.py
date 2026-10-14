"""Distributional parity of the pseudo bank vs the 5 published val streams (bit parity is impossible:
the public val plan is not reproducible from its seed for ADL pieces, see README).

Usage: python3 bank_stats.py  -> results/bank_stats.csv + printed table with KS tests
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
PSEUDO = Path("/home/user/data_raw/pseudo")
HERE = Path(__file__).resolve().parent


def stats(co: pd.DataFrame, ev: pd.DataFrame) -> tuple[dict, dict]:
    co = co.copy()
    for c in ("t_start", "t_end", "source_t0", "source_t1"):
        co[c] = co[c].astype(float)
    hours = co.groupby("stream_id").t_end.max().sum() / 3600
    fl = (ev.quality == "low") | ev.needs_review.astype(str).str.lower().eq("true")
    f = co[co.kind == "fall"].merge(ev[["source_path", "source_t_start", "source_t_end"]], left_on="path", right_on="source_path")
    pre = f.source_t_start.astype(float) - f.source_t0
    post = f.source_t1 - f.source_t_end.astype(float)
    adl = (co[co.kind == "adl"].source_t1 - co[co.kind == "adl"].source_t0)
    s = {
        "n_streams": co.stream_id.nunique(), "hours": hours, "falls": len(ev), "falls_per_h": len(ev) / hours,
        "evaluable": int((~fl).sum()), "junctions_per_h": (len(co) - co.stream_id.nunique()) / hours,
        "adl_pieces": int((co.kind == "adl").sum()), "adl_share": adl.sum() / 3600 / hours,
        "pre_margin_med": pre.median(), "post_margin_med": post.median(), "adl_piece_med": adl.median(),
        "adl_piece_p10": adl.quantile(0.1), "adl_piece_p90": adl.quantile(0.9),
    }
    return s, {"pre": pre.values, "post": post.values, "adl": adl.values}


def main() -> None:
    co = pd.read_csv(PKG / "streams_composition.csv", dtype={"dataset": str})
    ev = pd.read_csv(PKG / "streams_events.csv")
    real, rd = stats(co[co.stream_id.str.startswith("val")], ev[ev.stream_id.str.startswith("val")])
    rows = [{"split": "real_val", **real}]
    pooled = {"pre": [], "post": [], "adl": []}
    for p in sorted(PSEUDO.glob("events_*.csv")):
        seed = p.stem.split("_")[1]
        s, d = stats(pd.read_csv(PSEUDO / f"composition_{seed}.csv", dtype={"dataset": str}), pd.read_csv(p))
        rows.append({"split": f"seed_{seed}", **s})
        for k in pooled:
            pooled[k].append(d[k])
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "bank_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    for k in pooled:
        print(f"KS {k}: bank pooled vs real val p={ks_2samp(np.concatenate(pooled[k]), rd[k]).pvalue:.3f}")


if __name__ == "__main__":
    main()

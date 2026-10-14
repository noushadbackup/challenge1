"""Decode variants on the reference (G7N4) pipeline, per (bag, target):
  probabilities: G7-gated 0.75 TCN + 0.25 HGB, HGB optionally replaced by cache/hgb_<H>_<t>.npz (hgbvar.py)
  operating point: re-tuned on the variant's own gated train OOF (official evaluator, uwbfall grid, first max)
  boundaries: replay (row cache from dump.py; missing intervals replayed) -> refine with boundary overrides
  final: time NMS 4 s (reference)
With no option it must reproduce REF (= I1 G7N4) byte-for-byte (checked by --check).

    python3 work/I2_boundaries/decvar.py <targets> <name> [--hgb Z] [--agg cluster --cluster-iou 0.8] [--bags ...]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BAGS, COLS, HERE, REF_CFG, Probs, oname, target_sources  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall.boundary import refine, replay_stream, to_preds  # noqa: E402
from uwbfall.decode import events_df, to_events, tune  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from uwbfall.models import setup_torch  # noqa: E402
from uwbfall.scoring import gt_records  # noqa: E402
from uwbfall.sources import events_for  # noqa: E402

KEY = ["stream_id", "t_start", "t_end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("name")
    ap.add_argument("--hgb", default=None)
    ap.add_argument("--tcn", default=None, help="TCN bag name from tcnw.py (cache/probs_<name>_<t>.npz)")
    ap.add_argument("--agg", default="median")
    ap.add_argument("--cluster-iou", type=float, default=0.8)
    ap.add_argument("--fallback", default="detector")
    ap.add_argument("--contain-frac", type=float, default=0.8)
    ap.add_argument("--bags", default=",".join(BAGS))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cfg = REF_CFG
    bnd = {**cfg["boundary"], "agg": a.agg, "cluster_iou": a.cluster_iou, "fallback": a.fallback,
           "contain_frac": a.contain_frac}
    _lim = setup_torch(cfg)  # noqa: F841
    for t in a.targets.split(","):
        t0 = time.time()
        tr, evtr, te, _ = target_sources(t)
        te = sorted(te, key=lambda s: s.sid)
        gt = gt_records(events_for(tr, evtr), ignore_flagged=True)
        dur_tr = {s.sid: s.duration for s in tr}
        dur = {s.sid: s.duration for s in te}
        hgb = dict(np.load(HERE / "cache" / f"hgb_{a.hgb}_{t}.npz")) if a.hgb else None
        tcn = None
        if a.tcn:
            tcn = dict(np.load(HERE / "cache" / f"probs_{a.tcn}_{t}.npz"))
            from common import tdir
            sid_w = np.load(Path("/home/user/data_raw/I2/runs") / a.tcn / tdir(t) / "oof.npz")["sid"]
        rf = HERE / "cache" / f"rep_{t}_test.pkl"
        R = pickle.load(open(rf, "rb"))
        D = R["D"]
        ops, cands = {}, {}
        for bag in a.bags.split(","):
            P = Probs(bag, t, hgb=hgb, tcn=tcn)
            if tcn is not None:
                assert (P.sid == sid_w).all(), "TCN-W bundle block order differs"
            sc, g, _ = tune(P.oof(), gt, dur_tr, cfg)
            ops[bag] = {"op": list(g), "oof_rank_f1": sc}
            cands[bag] = events_df(sum((to_events(s.sid, P.test(s.sid), *g) for s in te), []))
        allc = pd.concat(cands.values()).drop_duplicates(KEY)
        miss = allc.merge(D[KEY], on=KEY, how="left", indicator=True)
        miss = miss[miss._merge == "left_only"].drop(columns="_merge")
        if len(miss):
            new = []
            for s in te:
                c = miss[miss.stream_id == s.sid].sort_values(KEY).reset_index(drop=True)
                if len(c):
                    new.append(replay_stream(s, get_features(s, cfg)["prof_02"], c, cfg).drop(columns=["score"]))
                    s.release()
            D = pd.concat([D] + new, ignore_index=True)
            R["D"] = D
            pickle.dump(R, open(rf, "wb"))
        for bag in a.bags.split(","):
            M = cands[bag].merge(D, on=KEY, how="left")
            s, e, _u = refine(M, bnd)
            pr = CL.nms_time(to_preds(M, s, e, dur), cfg["decode"]["nms_gap_s"]).reset_index(drop=True)[COLS]
            d = HERE / "preds" / bag / a.name
            d.mkdir(parents=True, exist_ok=True)
            pr.to_csv(d / f"{oname(t)}.csv", index=False)
            if a.check:
                ref = (HERE / "preds" / bag / "REF" / f"{oname(t)}.csv").read_bytes()
                print(f"  check {bag} {t}: identical to REF = {(d / f'{oname(t)}.csv').read_bytes() == ref}")
        f = HERE / "out" / f"ops_{a.name}.json"
        old = json.loads(f.read_text()) if f.exists() else {}
        old.update({f"{b}|{t}": v for b, v in ops.items()})
        f.write_text(json.dumps(old, indent=1))
        print(f"{a.name} {t}: ops {json.dumps({b: v['op'] for b, v in ops.items()})}, {len(miss)} new replays "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

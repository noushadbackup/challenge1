"""(c) Ignore-zone audit. For each bag x target (SR-CV test parts and official val):
  1. free rows: final rows that overlap an ignore zone (+-2 s, official rule) and hit no evaluable fall @0.1;
     check with the official evaluator that removing them leaves every count and the ranking F1 unchanged.
  2. NMS: pre-NMS rows that would hit an evaluable fall @0.3 but are suppressed by a higher-scored row that is
     itself a free ignore-zone row (the only way an ignore-zone row could cost us).
  3. per variant: rows the variant drops relative to REF, split into TP@0.3 / FP / free.

    python3 work/I3_score/ign.py V1,V2,...
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "I2_boundaries"))
from common import BAGS, oname, target_sources  # noqa: E402
from run import nms  # noqa: E402
from uwbfall import hitmodel as HM  # noqa: E402
from uwbfall.scoring import gt_records, score_inproc, summary  # noqa: E402
from uwbfall.sources import events_for  # noqa: E402

TARGETS = ["0", "1", "2", "3", "official"]


def lab(df, gt):
    return HM.hit_labels(df.stream_id.values, df.t_start.values, df.t_end.values, gt)


def main():
    V = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    out = {"free_rows": 0, "free_removal_changes_score": 0, "nms_suppressed_by_free": 0, "nms_suppressed_hits": 0,
           "dropped": {v: {"tp": 0, "fp": 0, "free": 0, "other": 0} for v in V}, "cells": 0}
    for t in TARGETS:
        _tr, _e, te, evte = target_sources(t)
        gt = gt_records(events_for(te, evte), ignore_flagged=True)
        dur = {s.sid: s.duration for s in te}
        for bag in BAGS:
            out["cells"] += 1
            ref = pd.read_csv(HERE / "preds" / bag / "REF" / f"{oname(t)}.csv")
            L = lab(ref, gt)
            free = (L.ignore_overlap & (L.best_iou < 0.1)).values
            out["free_rows"] += int(free.sum())
            s1 = summary(score_inproc(ref, gt, dur))
            s2 = summary(score_inproc(ref[~free], gt, dur))
            out["free_removal_changes_score"] += int(abs(s1["rank_f1"] - s2["rank_f1"]) > 1e-12)
            # NMS suppression by free rows
            F = pickle.load(open(HERE / "cache" / f"feat_{bag}_{t}.pkl", "rb"))
            pre = F["rows_te"].drop(columns="row").reset_index(drop=True)
            Lp = lab(pre, gt)
            kept = nms(pre.assign(_i=np.arange(len(pre))))._i.values
            sup = np.setdiff1d(np.arange(len(pre)), kept)
            for i in sup:
                if Lp["hit0.3"].iloc[i] != 1:
                    continue
                out["nms_suppressed_hits"] += 1
                r = pre.iloc[i]
                for k in kept:
                    q = pre.iloc[k]
                    if q.stream_id == r.stream_id and q.score >= r.score and \
                            max(r.t_start - q.t_end, q.t_start - r.t_end) < 4.0:
                        if Lp.ignore_overlap.iloc[k] and Lp.best_iou.iloc[k] < 0.1:
                            out["nms_suppressed_by_free"] += 1
                        break
            for v in V:
                f = HERE / "preds" / bag / v / f"{oname(t)}.csv"
                if not f.exists():
                    continue
                pv = pd.read_csv(f)
                m = ref.merge(pv[["stream_id", "t_start", "t_end"]], how="left", indicator=True,
                              on=["stream_id", "t_start", "t_end"])
                d = (m._merge == "left_only").values
                Ld = L[d]
                out["dropped"][v]["tp"] += int((Ld["hit0.3"] == 1).sum())
                fr = (Ld.ignore_overlap & (Ld.best_iou < 0.1))
                out["dropped"][v]["free"] += int(fr.sum())
                out["dropped"][v]["fp"] += int(((Ld["hit0.1"] == 0) & ~fr).sum())
                out["dropped"][v]["other"] += int(((Ld["hit0.1"] == 1) & (Ld["hit0.3"] == 0)).sum())
    (HERE / "out" / "ignore_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

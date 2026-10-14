"""Error anatomy of a prediction set on SR-CV folds (official matching @theta): where do FPs and FNs sit?

    python3 work/I1_clip/erran.py <pred_dir_with_fold_k.csv> [theta]
FP classes: in a fall clip whose GT is matched (fragment / post-fall / pre-fall), in a fall clip whose GT is unmatched
(mislocalised pair), in an ADL piece, in Autre/Empty, near a junction (<=2 s).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from srcv import core as C, scoring as S
from abc2027 import evaluate as E


def anatomy(pred_dir, theta=0.3, verbose=False):
    cnt = Counter()
    rows = []
    for k in range(4):
        dur = C.durations(k, "test")
        recs = S._records(pd.read_csv(Path(pred_dir) / f"fold_{k}.csv"), dur)
        gt, _ = S._gt([C.events_csv(k, "test")])
        comp = C.composition(k, "test")
        for sid in sorted(dur):
            cs = comp[comp.stream_id == sid].sort_values("t_start")
            g_s = [g for g in gt if g["stream_id"] == sid]
            p_s = [p for p in recs if p["stream_id"] == sid]
            res = E._match_stream(g_s, p_s, theta)
            mg = {gi for _, gi, _ in res["pairs"]}
            mp = {pi for pi, _, _ in res["pairs"]}
            scored = res["scored"]
            ign = [ev for ev in g_s if ev["ignore"]]
            for pi, p in enumerate(p_s):
                if pi in mp:
                    continue
                if any(E._intervals_overlap(p["t_start"], p["t_end"], ev["t_start"] - 2, ev["t_end"] + 2) for ev in ign):
                    continue
                m = 0.5 * (p["t_start"] + p["t_end"])
                ov = np.minimum(cs.t_end.values, p["t_end"]) - np.maximum(cs.t_start.values, p["t_start"])
                seg = cs.iloc[int(np.argmax(ov))]
                jd = np.min(np.abs(cs.t_start.values[1:] - p["t_start"])) if len(cs) > 1 else 99
                if seg.kind == "fall":
                    gi = [i for i, g in enumerate(scored) if seg.t_start - .3 <= g["t_start"] <= seg.t_end]
                    if gi:
                        g = scored[gi[0]]
                        rel = "post" if p["t_start"] >= g["t_end"] else ("pre" if p["t_end"] <= g["t_start"] else "overlap")
                        cls = f"fallclip_{'matched' if gi[0] in mg else 'unmatched'}_{rel}"
                    else:
                        cls = "fallclip_flaggedGT?"
                else:
                    cls = "adl" if seg.kind == "adl" else "autre_empty"
                if jd <= 2 and seg.kind != "fall":
                    cls += "_nearjunction"
                cnt["FP_" + cls] += 1
                rows.append((k, sid, p["t_start"], p["t_end"], p["score"], cls, seg.label))
            for gi, g in enumerate(scored):
                if gi in mg:
                    continue
                near = [p for p in p_s if E._intervals_overlap(p["t_start"], p["t_end"], g["t_start"] - 3, g["t_end"] + 3)]
                cnt["FN_" + ("has_pred_nearby" if near else "no_pred")] += 1
    return cnt, pd.DataFrame(rows, columns=["fold", "sid", "t0", "t1", "score", "cls", "label"])


if __name__ == "__main__":
    th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    for d in sys.argv[1].split(","):
        cnt, df = anatomy(d, th)
        print(d, dict(sorted(cnt.items())))
        if len(sys.argv) > 3:
            print(df.to_string())

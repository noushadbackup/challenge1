"""Voting / fusion schemes over the saved members. Everything is chosen on train OOF only
(leave-streams-out GroupKFold(4), official evaluator code in-process); each reported candidate is then
scored once on val with the official CLI.

Usage: python3 combine.py            -> out/combine_results.json + out/val_<name>.csv/json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from e5lib import (CACHE, GRID, OUT, STEP, events_from_probs, load, official, rank, score_inproc, summary, tune, write_sub)


RANK_GRID = [(thr, sm, md, gp) for thr in (0.97, 0.975, 0.98, 0.985, 0.99, 0.995) for sm in (1, 3, 5) for md in (0.4, 0.8) for gp in (0.2, 0.6)]


def fname(name):
    return name.replace(":", "__").replace("(", "").replace(")", "").replace(",", "_").replace("+", "p")

D = load()
TR_SID, TR_T, VA_SID, VA_T = D["train_sid"], D["train_t"], D["val_sid"], D["val_t"]
Y, FOLD = D["train_y"].astype(int), D["train_fold"]


def member(name):
    p = CACHE / "probs_tcn_e4.npz" if name == "tcn_e4" else OUT / f"probs_{name}.npz"
    with np.load(p) as z:
        return z["oof"].astype(np.float64), z["val"].astype(np.float64)


def lg(p):
    return logit(np.clip(p, 1e-5, 1 - 1e-5))


def qmap(oof, val):
    """Empirical-CDF (rank) transform; val mapped through the OOF CDF so a threshold transfers."""
    ref = np.sort(oof)
    return np.searchsorted(ref, oof, side="right") / len(ref), np.searchsorted(ref, val, side="right") / len(ref)


def mean_of(names, how="prob", w=None):
    P = [member(n) for n in names]
    w = np.ones(len(P)) / len(P) if w is None else np.asarray(w, float)
    if how == "prob":
        return sum(wi * o for wi, (o, _) in zip(w, P)), sum(wi * v for wi, (_, v) in zip(w, P))
    if how == "logit":
        return (expit(sum(wi * lg(o) for wi, (o, _) in zip(w, P))), expit(sum(wi * lg(v) for wi, (_, v) in zip(w, P))))
    if how == "rank":
        Q = [qmap(o, v) for o, v in P]
        return sum(wi * o for wi, (o, _) in zip(w, Q)), sum(wi * v for wi, (_, v) in zip(w, Q))
    raise ValueError(how)


def kth_largest(names, k):
    O = np.sort(np.stack([member(n)[0] for n in names]), axis=0)[::-1]
    V = np.sort(np.stack([member(n)[1] for n in names]), axis=0)[::-1]
    return O[k - 1], V[k - 1]


def stack(names):
    """Logistic meta-model on member logits (+ their +/-1 s smoothed context). OOF via cross-fitting on the same
    stream folds (meta for fold k is fit on the OOF of the other folds); val via the meta fit on all OOF."""
    def feats(cols, sid):
        Z = np.stack([lg(c) for c in cols], 1)
        out = [Z]
        sm = np.empty_like(Z)
        for s in np.unique(sid):
            idx = np.flatnonzero(sid == s)
            for j in range(Z.shape[1]):
                sm[idx, j] = np.convolve(Z[idx, j], np.ones(11) / 11, "same")
        out.append(sm)
        return np.hstack(out)
    P = [member(n) for n in names]
    Xo, Xv = feats([o for o, _ in P], TR_SID), feats([v for _, v in P], VA_SID)
    oof = np.zeros(len(Y))
    for k in range(4):
        a, b = FOLD != k, FOLD == k
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xo[a], Y[a])
        oof[b] = m.predict_proba(Xo[b])[:, 1]
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Xo, Y)
    coef = dict(zip([f"{n}" for n in names] + [f"{n}~sm" for n in names], np.round(m.coef_[0], 3).tolist()))
    return oof, m.predict_proba(Xv)[:, 1], coef


# ---------------- event-level fusion -----------------
def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]) + 1e-9)


def member_events(names, split, cfgs):
    out = {}
    for n in names:
        o, v = member(n)
        sid, t, p = (TR_SID, TR_T, o) if split == "train" else (VA_SID, VA_T, v)
        out[n] = events_from_probs(sid, t, p, cfgs[n])
    return out


def wbf(evd: dict, min_support: int, min_score: float, iou_thr=0.1):
    """1-D weighted segment fusion: cluster events across members by tIoU with the running fused box;
    fused box = score-weighted mean of start/end; fused score = sum(scores)/n_members."""
    n_m = len(evd)
    pool = [(e[0], e[1], e[2], e[3], m) for m, evs in evd.items() for e in evs]
    out = []
    for s in sorted({e[0] for e in pool}):
        evs = sorted([e for e in pool if e[0] == s], key=lambda e: -e[3])
        clusters = []  # [box(start,end), list of events]
        for e in evs:
            best, bi = 0.0, -1
            for i, c in enumerate(clusters):
                v = tiou((e[1], e[2]), c[0])
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= iou_thr:
                c = clusters[bi]
                c[1].append(e)
                w = np.array([x[3] for x in c[1]])
                c[0] = (float(np.dot(w, [x[1] for x in c[1]]) / w.sum()), float(np.dot(w, [x[2] for x in c[1]]) / w.sum()))
            else:
                clusters.append([(e[1], e[2]), [e]])
        for box, mem in clusters:
            support = len({x[4] for x in mem})
            # one score per member (best) so duplicates inside a member do not inflate the score
            best_per = {}
            for x in mem:
                best_per[x[4]] = max(best_per.get(x[4], 0), x[3])
            sc = sum(best_per.values()) / n_m
            if support >= min_support and sc >= min_score:
                out.append((s, box[0], box[1], sc, box[0]))
    return out


def soft_nms(evd: dict, min_score: float, sigma=0.5):
    pool = [(e[0], e[1], e[2], e[3]) for evs in evd.values() for e in evs]
    out = []
    for s in sorted({e[0] for e in pool}):
        evs = [list(e) for e in pool if e[0] == s]
        keep = []
        while evs:
            i = int(np.argmax([e[3] for e in evs]))
            top = evs.pop(i)
            keep.append(top)
            for e in evs:
                e[3] *= np.exp(-tiou((e[1], e[2]), (top[1], top[2])) ** 2 / sigma)
        out += [(e[0], e[1], e[2], e[3], e[1]) for e in keep if e[3] >= min_score]
    return out


def main():
    t0 = time.time()
    have = [p.stem.replace("probs_", "") for p in OUT.glob("probs_*.npz")] + ["tcn_e4"]
    print("members:", have, flush=True)
    committee = [m for m in ("lgb_fused", "xgb_fused", "cat_fused", "hgb_fused") if m in have]
    radars = [m for m in ("lgb_radar0", "lgb_radar1", "lgb_radar2") if m in have]
    cands: dict[str, tuple] = {}
    # (a) singles
    for m in have:
        cands[f"single:{m}"] = member(m)
    # (b) model committee
    cands["committee:mean"] = mean_of(committee)
    cands["committee:logit_mean"] = mean_of(committee, "logit")
    cands["committee:rank_mean"] = mean_of(committee, "rank")
    # (a) radar agents
    if len(radars) == 3:
        cands["radar:mean3"] = mean_of(radars)
        cands["radar:max"] = kth_largest(radars, 1)
        cands["radar:2of3_agree"] = kth_largest(radars, 2)
        o2, v2 = kth_largest(radars, 2)
        of, vf = member("lgb_fused")
        cands["radar:fused_OR_2of3"] = (np.maximum(of, o2), np.maximum(vf, v2))
        omx, vmx = kth_largest(radars, 1)
        cands["radar:mean(fused,max_radar)"] = ((of + omx) / 2, (vf + vmx) / 2)
        cands["radar:mean(fused,3radars)"] = mean_of(["lgb_fused"] + radars)
    # (e) physics hedge blended
    if "lgb_physics" in have:
        oc, vc = cands["committee:mean"]
        op, vp = member("lgb_physics")
        for w in (0.2, 0.33):
            cands[f"hedge:committee+physics_w{w}"] = ((1 - w) * oc + w * op, (1 - w) * vc + w * vp)
    # (c) stacking
    coefs = {}
    stack_sets = {"stack:committee": committee, "stack:committee+radars+physics": committee + radars + ["lgb_physics"]}
    stack_sets["stack:all+tcn"] = committee + radars + ["lgb_physics", "tcn_e4"]
    for name, names in stack_sets.items():
        o, v, c = stack(names)
        cands[name], coefs[name] = (o, v), c
    # TCN (E4 output, reused) in the committee
    for wt in (0.25, 0.5, 0.75):
        cands[f"committee+tcn:w_tcn{wt}"] = mean_of(committee + ["tcn_e4"], w=[(1 - wt) / 4] * 4 + [wt])
    cands["committee+tcn:mean_equal5"] = mean_of(committee + ["tcn_e4"])

    results = {}
    cfgs = {}
    for name, (o, v) in cands.items():
        # rank-mean lives on the CDF scale (falls ~1.6% of blocks): thresholds must be upper quantiles
        grid = RANK_GRID if "rank" in name else GRID
        sc, g = tune(TR_SID, TR_T, o, grid=grid)
        cfgs[name] = g
        write_sub(events_from_probs(TR_SID, TR_T, o, g), OUT / f"oof_{fname(name)}.csv")
        results[name] = {"train_oof_rank_f1": round(sc, 4), "postproc": g, "val_probs": v}
        print(f"{name:40s} oof={sc:.4f} {g}  [{time.time() - t0:.0f}s]", flush=True)

    # (d) event-level fusion over the committee (+ radar agents), each member with its own tuned post-processing
    single_cfg = {m: cfgs[f"single:{m}"] for m in have}
    ev_sets = {"committee": committee, "committee+radars": committee + radars, "committee+tcn": committee + ["tcn_e4"]}
    ev_results = {}
    for setname, names in ev_sets.items():
        tr_ev = member_events(names, "train", single_cfg)
        best = None
        for ms in range(1, len(names) + 1):
            for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
                sc = rank(score_inproc(wbf(tr_ev, ms, thr), "train"))
                if best is None or sc > best[0] + 1e-12:
                    best = (sc, (ms, thr))
        write_sub(wbf(tr_ev, *best[1]), OUT / f"oof_{fname('wbf1d:' + setname)}.csv")
        ev_results[f"wbf1d:{setname}"] = {"train_oof_rank_f1": round(best[0], 4), "postproc": {"min_support": best[1][0], "min_score": best[1][1]}, "names": names, "kind": "wbf", "cfg": best[1]}
        print(f"wbf1d:{setname:30s} oof={best[0]:.4f} {best[1]}", flush=True)
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            sc = rank(score_inproc(soft_nms(tr_ev, thr), "train"))
            if best is None or sc > best[0] + 1e-12:
                best = (sc, thr)
        write_sub(soft_nms(tr_ev, best[1]), OUT / f"oof_{fname('softnms:' + setname)}.csv")
        ev_results[f"softnms:{setname}"] = {"train_oof_rank_f1": round(best[0], 4), "postproc": {"min_score": best[1]}, "names": names, "kind": "softnms", "cfg": best[1]}
        print(f"softnms:{setname:30s} oof={best[0]:.4f} {best[1]}", flush=True)

    # ---- single val scoring (official CLI) for every candidate ----
    final = {}
    for name, r in results.items():
        evs = events_from_probs(VA_SID, VA_T, r["val_probs"], r["postproc"])
        rv = official(evs, "val_" + fname(name))
        final[name] = {"train_oof_rank_f1": r["train_oof_rank_f1"], "postproc": r["postproc"], "val": summary(rv)}
    for name, r in ev_results.items():
        va_ev = member_events(r["names"], "val", single_cfg)
        evs = wbf(va_ev, *r["cfg"]) if r["kind"] == "wbf" else soft_nms(va_ev, r["cfg"])
        rv = official(evs, "val_" + fname(name))
        final[name] = {"train_oof_rank_f1": r["train_oof_rank_f1"], "postproc": r["postproc"], "val": summary(rv)}
    final["_stack_coefs"] = coefs
    json.dump(final, open(OUT / "combine_results.json", "w"), indent=1)
    print("\n%-42s %8s %8s  %s" % ("candidate", "OOF", "VAL", "val f1@.1/.3/.5/.7 | tp/fp/fn | med start err"))
    for name, r in sorted(((k, v) for k, v in final.items() if not k.startswith("_")), key=lambda kv: -kv[1]["train_oof_rank_f1"]):
        v = r["val"]
        print("%-42s %8.4f %8.4f  %s | %s | %s" % (name, r["train_oof_rank_f1"], v["rank_f1"], v["f1@0.1/0.3/0.5/0.7"], v["tp_fp_fn@0.3"], v["start_err_med@0.3"]))
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

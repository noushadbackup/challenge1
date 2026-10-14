"""E3: does the complex I/Q physics feature group help the prototype 'full' detector?

Usage: python3 iq_experiment.py [--val] [--seeds 0] [--es] arm [arm ...]
  arms: full, full+iq, full+zone, full+iq+zone, full+iq+zone+seg, physics, physics+iq+zone, ...
        (any '+'-joined combination of base set {physics,structure,full} and groups {iq,zone,seg})

Protocol (identical to analysis/prototype_detector.py except where noted):
  * 0.2 s blocks, HistGradientBoosting with the prototype hyper-parameters.
  * Train OOF via leave-streams-out GroupKFold(4) over the 13 train streams (R repeats with
    shuffled stream->fold maps, --reps), post-processing grid tuned on train OOF only, scored
    with the official evaluator functions (abc2027.evaluate.evaluate, imported in-process).
  * Default: early_stopping=False (deterministic; X5 showed 'auto' early stopping makes the
    prototype seed-dependent). --es restores early_stopping='auto' with random_state=seed.
  * --val: refit on all train, predict the 5 val streams once, write CSV, score with the
    official CLI (--ignore-flagged).
Outputs go to results/<arm>[_es_sSEED].json and preds/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/challenge1/analysis")
sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

import prototype_detector as P  # noqa: E402
from abc2027 import evaluate as EV  # noqa: E402
from common import FEATS, PKG, stream_ids  # noqa: E402

IQ = Path("/home/user/data_raw/feats_iq")
ZONES = {0: (40, 69), 1: (5, 22), 2: (5, 22)}
STEP = P.STEP
EPS = 1e-9


def roll_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Centred rolling sum over k blocks along the last axis (zero padded)."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x.astype(np.float64), k, axis=-1, mode="constant") * k


def lg(x):
    return np.log10(np.maximum(x, 0) + 1e-6)


def iq_features(sid: str, n: int) -> dict[str, np.ndarray]:
    """Block-level complex I/Q physics features (per radar + max over radars)."""
    z = dict(np.load(IQ / f"{sid}.npz"))
    z = {k: v[:, :n].astype(np.float64) for k, v in z.items()}
    f: dict[str, np.ndarray] = {}
    cnt = np.maximum(z["cnt"], 1)
    f_r: dict[str, np.ndarray] = {}
    f_r["hold"] = 1 - z["cnt"] / 10
    for sc, k in (("02", 1), ("1", 5)):
        S = (lambda a: a) if k == 1 else (lambda a, k=k: roll_sum(a, k))
        P_ = np.hypot(S(z["P_re"]), S(z["P_im"]))
        coh = P_ / np.sqrt(S(z["E0"]) * S(z["E1"]) + EPS)
        f_r[f"incoh{sc}"] = lg(1 - np.clip(coh, 0, 1))  # raw lag-1 decorrelation (0 when static)
        eh = S(z["Eh"]) / np.maximum(S(cnt), 1)
        f_r[f"hpE{sc}"] = lg(eh / (np.median(eh, axis=1, keepdims=True) + EPS))
        Q_ = np.hypot(S(z["Q_re"]), S(z["Q_im"]))
        f_r[f"hpcoh{sc}"] = Q_ / (S(z["Sw"]) + EPS)  # phase concentration of moving part (1 - circ var)
        f_r[f"hishare{sc}"] = S(z["Swhi"]) / (S(z["Sw"]) + EPS)  # share of |dphi| > pi/2 (fast / aliased)
        md = S(z["Md"]) / 1.0
        f_r[f"md{sc}"] = lg(md / (np.median(md, axis=1, keepdims=True) + EPS))
        mz = S(z["Mz"])
        f_r[f"mz{sc}"] = lg(mz / (np.median(mz, axis=1, keepdims=True) + EPS))
        # signed mean Doppler (rad/s) of the moving part, weighted by energy
        ang = np.arctan2(S(z["Q_im"]), S(z["Q_re"]))
        dt = S(z["sdt"]) / np.maximum(S(cnt), 1)
        f_r[f"dopp{sc}"] = ang / np.maximum(dt, 0.02)
        U = np.hypot(S(z["U_re"]), S(z["U_im"]))
        f_r[f"ucoh{sc}"] = U / np.maximum(S(cnt), 1)
        c0 = S(z["Cb0"]) + EPS
        cen = S(z["Cb1"]) / c0
        spr = np.sqrt(np.maximum(S(z["Cb2"]) / c0 - cen**2, 0))
        lo = np.array([ZONES[r][0] for r in range(3)])[:, None]
        f_r[f"cen{sc}"] = cen - lo
        f_r[f"spread{sc}"] = spr
    # range-centroid velocity (bins per s -> x 5.14 cm) over +-1 s on the 1 s centroid
    c1 = f_r["cen1"]
    f_r["cenvel"] = (P.shift(c1, 5) - P.shift(c1, -5)) / 2.0 * 0.0514
    for key, v in f_r.items():
        for r in range(3):
            f[f"iq_{key}_r{r}"] = v[r].astype(np.float32)
        if key in ("hpE02", "hpE1", "md02", "md1", "mz02", "mz1", "incoh02", "incoh1"):
            f[f"iq_{key}_max"] = v.max(0).astype(np.float32)
    for key in ("iq_hpE1_max", "iq_mz1_max", "iq_incoh1_max"):
        for k in (-10, -5, 5, 10):
            f[f"{key}@{k * STEP:+.0f}s"] = P.shift(f[key], k)
    return f


def zone_static(sid: str, n: int) -> dict[str, np.ndarray]:
    """Static past-vs-future change restricted to each radar's mat range bins (cached prof_02)."""
    F = np.load(FEATS / f"{sid}.npz")
    p = F["prof_02"].astype(np.float32) / 1e3
    f = {}
    chs = {5: [], 10: [], 25: []}
    for r in range(3):
        a, b = ZONES[r]
        zz = (p[r, :, a:b] + 1j * p[r, :, 184 + a:184 + b]).T  # (B, n)
        norm = np.median(np.linalg.norm(zz, axis=0)) + 1e-12
        for k in chs:
            m = P.rolling_mean(zz, k)
            past = np.concatenate([np.repeat(m[..., :1], k, axis=-1), m], axis=-1)[..., :n]
            fut = np.concatenate([m[..., 1:], np.repeat(m[..., -1:], k, axis=-1)], axis=-1)[..., :n]
            ch = np.log10(np.linalg.norm(fut - past, axis=0) / norm + 1e-6)
            chs[k].append(ch)
            f[f"zst{k}_r{r}"] = ch.astype(np.float32)
    for k in chs:
        f[f"zst{k}_max"] = np.max(chs[k], axis=0).astype(np.float32)
    for k in (-10, -5, 5, 10):
        f[f"zst10_max@{k * STEP:+.0f}s"] = P.shift(f["zst10_max"], k)
    return f


def seg_features(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Segment-relative features over the prototype's jump-to-jump segment (_prev id)."""
    g = d.groupby("_prev")
    out = {}
    for key in ("zst10_max", "iq_hpE1_max", "iq_mz1_max"):
        if key not in d:
            continue
        smax = g[key].transform("max").values
        out[f"sg_{key}_rel"] = d[key].values - smax
        # time to the segment argmax of this signal
        idx = np.arange(len(d))
        am = d.assign(_i=idx).loc[g[key].idxmax().values, ["_prev", "_i"]].set_index("_prev")["_i"]
        out[f"sg_{key}_dt"] = np.clip((idx - am.reindex(d["_prev"]).values) * STEP, -30, 30)
    if "iq_hold_r0" in d:
        out["sg_hold_mean"] = g["iq_hold_r0"].transform("mean").values
    return out


def build_all(sids, groups):
    frames = P.build(sids)
    out = []
    for d, amp in frames:
        sid = d.sid.iloc[0]
        n = len(d)
        extra = {}
        if "iq" in groups or "seg" in groups:
            extra.update(iq_features(sid, n))
        if "zone" in groups or "seg" in groups:
            extra.update(zone_static(sid, n))
        d = pd.concat([d, pd.DataFrame(extra, index=d.index)], axis=1)
        if "seg" in groups:
            d = pd.concat([d, pd.DataFrame(seg_features(d), index=d.index)], axis=1)
        out.append((d, amp))
    return out


def cols_for(df, base, groups):
    cols = P.cols_for(df, base)
    if "iq" in groups:
        cols += [c for c in df.columns if c.startswith("iq_")]
    if "zone" in groups:
        cols += [c for c in df.columns if c.startswith("zst")]
    if "seg" in groups:
        cols += [c for c in df.columns if c.startswith("sg_")]
    return list(dict.fromkeys(cols))


_GT = None


def gt():
    global _GT
    if _GT is None:
        _GT = EV.load_ground_truth(PKG / "streams_events.csv", ignore_flagged=True)
    return _GT


_DUR = {}


def durations(split):
    if split not in _DUR:
        _DUR[split] = EV.stream_durations(PKG / "streams" / split)
    return _DUR[split]


def to_preds(evs):
    return [{"stream_id": s, "t_start": a, "t_end": b, "score": sc, "t_alarm": al} for s, a, b, sc, al in evs]


def score(evs, split, sids=None):
    dur = durations(split)
    if sids is not None:
        dur = {k: v for k, v in dur.items() if k in sids}
    return EV.evaluate(gt(), to_preds(evs), dur)


GRID = [(thr, sm, md, gp) for thr in (0.3, 0.4, 0.5, 0.6, 0.7) for sm in (1, 3, 5) for md in (0.4, 0.8) for gp in (0.2, 0.6)]


def events_from(df, p, g):
    return [e for sid, d in df.assign(p=p).groupby("sid") for e in P.to_events(sid, d.t.values - STEP / 2, d.p.values, *g)]


def per_stream_counts(evs, split, sids):
    """tp/fp/fn per stream per theta (official matching), for the paired stream bootstrap."""
    rows = {}
    for s in sids:
        e = [x for x in evs if x[0] == s]
        r = score(e, split, {s})
        rows[s] = {th: (r["by_theta"][th]["tp"], r["by_theta"][th]["fp"], r["by_theta"][th]["fn"]) for th in ("0.1", "0.3", "0.5")}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--es", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "preds").mkdir(exist_ok=True)
    allg = set()
    for arm in a.arms:
        allg |= set(arm.split("+")[1:])
    sids = stream_ids()
    tr_ids = [s for s in sids if s.startswith("train")]
    va_ids = [s for s in sids if s.startswith("val")]
    t0 = time.time()
    tr, pca = P.add_fingerprint(build_all(tr_ids, allg), None)
    va, _ = P.add_fingerprint(build_all(va_ids, allg), pca) if a.val else (None, None)
    print(f"built features {time.time() - t0:.0f}s rows={len(tr)}", flush=True)
    for arm in a.arms:
        base, *groups = arm.split("+")
        cols = cols_for(tr, base, groups)
        for seed in a.seeds:
            tag = arm + (f"_es_s{seed}" if a.es else "") + (f"_r{a.reps}" if a.reps > 1 else "")
            params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed,
                          early_stopping="auto" if a.es else False)
            t1 = time.time()
            oofs = []
            for rep in range(a.reps):
                perm = np.random.default_rng(rep).permutation(len(tr_ids)) if rep else np.arange(len(tr_ids))
                fold_of = {s: perm[i] % 4 for i, s in enumerate(tr_ids)} if rep else None
                oof = np.zeros(len(tr))
                if rep == 0:
                    splits = list(P.GroupKFold(4).split(tr, tr.y, tr.sid))
                else:
                    fo = tr.sid.map(fold_of).values
                    splits = [(np.flatnonzero(fo != k), np.flatnonzero(fo == k)) for k in range(4)]
                for tri, tei in splits:
                    m = HistGradientBoostingClassifier(**params).fit(tr.iloc[tri][cols], tr.y.iloc[tri])
                    oof[tei] = m.predict_proba(tr.iloc[tei][cols])[:, 1]
                oofs.append(oof)
            # tune post-processing on the mean OOF score across repetitions
            res_grid = []
            for g in GRID:
                scs = [score(events_from(tr, o, g), "train")["ranking_score"]["f1_mean"] for o in oofs]
                res_grid.append((float(np.mean(scs)), g, scs))
            best = max(res_grid, key=lambda x: x[0])
            ev_oof = events_from(tr, oofs[0], best[1])
            r_oof = score(ev_oof, "train")
            res = {
                "arm": arm, "seed": seed, "early_stopping": a.es, "reps": a.reps, "n_features": len(cols),
                "train_oof_rank_f1": round(best[0], 4), "train_oof_rank_f1_per_rep": [round(s, 4) for s in best[2]],
                "postproc": best[1],
                "train_oof_f1@0.1/0.3/0.5/0.7": [round(r_oof["by_theta"][k]["f1"], 3) for k in ("0.1", "0.3", "0.5", "0.7")],
                "train_oof_tp_fp_fn@0.3": [r_oof["by_theta"]["0.3"][k] for k in ("tp", "fp", "fn")],
                "train_oof_per_stream": per_stream_counts(ev_oof, "train", tr_ids),
            }
            np.save(HERE / "preds" / f"oof_{tag}.npy", oofs[0])
            if a.val:
                model = HistGradientBoostingClassifier(**params).fit(tr[cols], tr.y)
                pv = model.predict_proba(va[cols])[:, 1]
                np.save(HERE / "preds" / f"valprob_{tag}.npy", pv)
                csv = HERE / "preds" / f"val_{tag}.csv"
                P.write_sub(events_from(va, pv, best[1]), csv)
                r = P.official(csv.resolve(), "val", (HERE / "preds" / f"val_{tag}.json").resolve())
                bt = r["by_theta"]
                res.update({
                    "val_rank_f1": round(r["ranking_score"]["f1_mean"], 4),
                    "val_f1@0.1/0.3/0.5/0.7": [round(bt[k]["f1"], 3) for k in ("0.1", "0.3", "0.5", "0.7")],
                    "val_tp_fp_fn@0.3": [bt["0.3"]["tp"], bt["0.3"]["fp"], bt["0.3"]["fn"]],
                    "val_fa_per_hour@0.3": round(bt["0.3"]["false_alarms_per_hour"], 2),
                    "val_start_err_median@0.3": round(bt["0.3"]["start_error_median_abs"] or -1, 3),
                    "val_mean_tiou@0.3": round(bt["0.3"]["mean_tiou"] or -1, 3),
                })
                if hasattr(model, "feature_names_in_"):
                    res["features"] = cols
            res["wall_s"] = round(time.time() - t1, 1)
            json.dump(res, open(HERE / "results" / f"{tag}.json", "w"), indent=1)
            print(tag, json.dumps({k: v for k, v in res.items() if k not in ("train_oof_per_stream", "features")}), flush=True)


if __name__ == "__main__":
    main()

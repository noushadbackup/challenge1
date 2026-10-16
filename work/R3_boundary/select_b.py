"""Use the dense boundary model (i) as fallback for gate-failed candidates and (ii) together with the replay
boundaries; every parameter is chosen on the target's TRAINING-OOF candidates (D_train + inner-OOF logits).

    python3 select_b.py <tag> [targets...]          -> results/<variant>/{fold_k,val}.csv + sel_<tag>.json
"""
from common import *  # noqa
import itertools
import json
from bmodel import window_logp, DT
from uwbfall.boundary import refine, to_preds
from uwbfall.pipeline import load_bundle, _nms
from uwbfall.scoring import gt_records, score_inproc, summary

W = 4.0


def cand_model(D: pd.DataFrame, L: dict, tol: float = 0.1):
    """Per candidate: model start/end log-prob tracks around the detector bounds."""
    out = []
    for r in D.itertuples():
        lg = L[r.stream_id].astype(np.float32)
        ts, lps = window_logp(lg[0], r.t_start, W)
        te, lpe = window_logp(lg[1], r.t_end, W)
        out.append((ts, lps, te, lpe))
    return out


def mass(t, lp, x, tol):
    """log P(|T - x| <= tol) under the window distribution (-inf outside the window)."""
    if not np.isfinite(x):
        return -np.inf
    m = np.abs(t - x) <= tol + 1e-9
    if not m.any():
        return -50.0
    v = lp[m]
    return float(np.log(np.sum(np.exp(v - v.max()))) + v.max())


def model_bounds(tr, dmin, how="argmax"):
    ts, lps, te, lpe = tr
    if how == "argmax":
        s = ts[np.argmax(lps)]; e = te[np.argmax(lpe)]
    else:   # local expectation around the mode (+-0.2 s)
        i = np.argmax(lps); j = np.argmax(lpe)
        ms = np.abs(ts - ts[i]) <= 0.2; me = np.abs(te - te[j]) <= 0.2
        ps = np.exp(lps[ms] - lps[ms].max()); pe = np.exp(lpe[me] - lpe[me].max())
        s = float(np.sum(ts[ms] * ps) / ps.sum()); e = float(np.sum(te[me] * pe) / pe.sum())
    e = max(e, s + dmin)
    return s, e


def bounds(D, b, M, v: dict):
    """v: {'fb': 'det'|'model'|'sel', 'm': margin (inf = keep median), 'pool_model': bool, 'lam': float, 'tol': float,
    'how': 'argmax'|'exp'}"""
    if "gate" in v:
        b = {**b, "gate": v["gate"]}
    a, e, use = refine(D, b)
    a, e = a.copy(), e.copy()
    s0, e0 = D.t_start.values.astype(float), D.t_end.values.astype(float)
    S = np.vstack([D[f"{c}_s"].values for c in b["crops"]]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in b["crops"]]).astype(float)
    dmin = b["dmin"]
    tol = v.get("tol", 0.1)
    for i in range(len(D)):
        tr = M[i] if M is not None else None
        if tr is not None:
            ms, me = model_bounds(tr, dmin, v.get("how", "argmax"))
        sc = lambda hs, he: mass(tr[0], tr[1], hs, tol) + mass(tr[2], tr[3], he, tol)
        H = [(S[c, i], max(E[c, i], S[c, i] + dmin)) for c in range(S.shape[0])
             if np.isfinite(S[c, i]) and np.isfinite(E[c, i]) and E[c, i] > S[c, i]]
        H.append((s0[i], max(e0[i], s0[i] + dmin)))
        if v.get("pool_model", False):
            H.append((ms, me))
        if not use[i]:
            fb = v.get("fb", "det")
            if fb == "model":
                a[i], e[i] = ms, me
            elif fb == "sel":
                k = int(np.argmax([sc(*h) for h in H]))
                a[i], e[i] = H[k]
        else:
            wds, wde = v.get("wdet_s", v.get("wdet", 0.0)), v.get("wdet_e", v.get("wdet", 0.0))
            if float(tiou(a[i], e[i], s0[i], e0[i])) >= v.get("tau", 9.0):   # replay and detector agree: keep replay
                wds = wde = 0.0
            lam = v.get("lam", 0.0)
            if wds > 0 or wde > 0 or lam > 0:
                if v.get("onesided") and s0[i] > a[i]:   # only move the start earlier
                    wds = 0.0
                ns = (1 - wds - lam) * a[i] + wds * s0[i] + (lam * ms if lam > 0 else 0.0)
                ne = (1 - wde - lam) * e[i] + wde * e0[i] + (lam * me if lam > 0 else 0.0)
                a[i], e[i] = ns, max(ne, ns + dmin)
            m = v.get("m", np.inf)
            if np.isfinite(m):
                scs = [sc(*h) for h in H]
                k = int(np.argmax(scs))
                if scs[k] > sc(a[i], e[i]) + m:
                    a[i], e[i] = H[k]
    return a, e, use


def final_rows(D, b, M, v, dur, cfg):
    a, e, _ = bounds(D, b, M, v)
    return _nms(to_preds(D, a, e, dur), cfg, {})


GRID = ([{"name": "base"}]
        + [{"name": f"fb_model_{h}", "fb": "model", "how": h} for h in ("argmax", "exp")]
        + [{"name": f"fb_sel_t{tol}_pm{int(pm)}", "fb": "sel", "tol": tol, "pool_model": pm}
           for tol in (0.1, 0.2) for pm in (False, True)]
        + [{"name": f"lam{l}", "lam": l} for l in (0.1, 0.2, 0.3, 0.4, 0.5)]
        + [{"name": f"wdet{l}", "wdet": l} for l in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)]
        + [{"name": f"gate{g}", "gate": g} for g in (0.2, 0.4, 0.5)]
        + [{"name": f"sel_m{m}_t{tol}_pm{int(pm)}", "m": m, "tol": tol, "pool_model": pm}
           for m in (0.0, 1.0, 2.0, 4.0) for tol in (0.1, 0.2) for pm in (False, True)])


def run_target(tag, t, grid=GRID):
    B = load_bundle(bundle_dir(t))
    cfg, b = B["cfg"], B["cfg"]["boundary"]
    Dt = pd.read_csv(DUMP / tname(t) / "D_train.csv")
    Mt = None if tag == "none" else cand_model(Dt, dict(np.load(BIG / "logits" / tag / tname(t) / "train.npz")))
    if Mt is None:
        grid = [v for v in grid if not any(k in v for k in ("fb", "lam", "m", "pool_model"))]
    dur = {s.sid: s.duration for s in train_sources(t)}
    gt = gt_records(train_events(t), ignore_flagged=True)
    res = {}
    for v in grid:
        r = final_rows(Dt, b, Mt, v, dur, cfg)
        res[v["name"]] = summary(score_inproc(r, gt, dur))
    return res


if __name__ == "__main__":
    tag = sys.argv[1]
    tg = [("val" if a == "val" else int(a)) for a in sys.argv[2:]] or TARGETS
    allr = {}
    for t in tg:
        allr[tname(t)] = run_target(tag, t)
        df = pd.DataFrame({k: {"rank": x["rank_f1"], **dict(zip(["f1", "f3", "f5", "f7"], x["f1@0.1/0.3/0.5/0.7"]))}
                           for k, x in allr[tname(t)].items()}).T
        print(f"== {tname(t)} (training OOF)\n{df.to_string()}", flush=True)
    (HERE / f"sel_train_{tag}.json").write_text(json.dumps(allr, indent=1))

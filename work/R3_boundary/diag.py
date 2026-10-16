"""Where do the tIoU-0.5/0.7 boundary errors come from? (gate-failed candidates vs agreeing crops)

For every evaluable GT fall: the candidate with the largest detector-bounds tIoU (>= 0.1 = 'detected'), its final
replay bounds (refine), whether any crop passed the agreement gate ('agree') or not ('gate_fail' -> detector bounds),
and oracle tIoUs (best single crop, best of crops + detector).

    python3 diag.py train|test [targets...]
Training OOF ('train') is the legal place for choices; 'test' is reported as a diagnostic only.
"""
from common import *  # noqa
import json
from uwbfall.boundary import refine
from uwbfall.pipeline import load_bundle


def per_event(D: pd.DataFrame, ev: pd.DataFrame, b: dict) -> pd.DataFrame:
    a, e, use = refine(D, b)
    s0, e0 = D.t_start.values, D.t_end.values
    S = np.vstack([D[f"{c}_s"].values for c in b["crops"]]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in b["crops"]]).astype(float)
    E = np.where(np.isfinite(E), np.maximum(E, S + b["dmin"]), np.nan)
    rows = []
    ev = unflagged(ev)
    for r in ev.itertuples():
        m = np.flatnonzero(D.stream_id.values == r.stream_id)
        gs, ge = float(r.t_start), float(r.t_end)
        if not len(m):
            rows.append(dict(stream_id=r.stream_id, gs=gs, ge=ge, fall_type=getattr(r, "fall_type", ""), detected=False))
            continue
        td = tiou(s0[m], e0[m], gs, ge)
        j = m[int(np.argmax(td))]
        if td.max() < 0.1:
            rows.append(dict(stream_id=r.stream_id, gs=gs, ge=ge, fall_type=getattr(r, "fall_type", ""), detected=False))
            continue
        tc = tiou(S[:, j], E[:, j], gs, ge)
        tc = np.where(np.isfinite(tc), tc, 0)
        rows.append(dict(stream_id=r.stream_id, gs=gs, ge=ge, fall_type=getattr(r, "fall_type", ""), detected=True,
                         row=j, agree=bool(use[j]), t_final=float(tiou(a[j], e[j], gs, ge)),
                         t_det=float(td.max()), t_crop_best=float(tc.max()),
                         t_oracle=float(max(tc.max(), tiou(s0[j], max(e0[j], s0[j] + b["dmin"]), gs, ge))),
                         err_s=a[j] - gs, err_e=e[j] - ge, dur=ge - gs))
    return pd.DataFrame(rows)


def table(P: pd.DataFrame) -> dict:
    out = {"n_eval": int(len(P)), "det_miss": int((~P.detected.astype(bool)).sum())}
    d = P[P.detected.astype(bool)].copy(); d["agree"] = d.agree.astype(bool)
    for th in (0.5, 0.7):
        for grp, g in (("agree", d[d.agree]), ("gate_fail", d[~d.agree])):
            out[f"{grp}_n"] = int(len(g))
            out[f"{grp}_fail@{th}"] = int((g.t_final < th).sum())
            out[f"{grp}_oracle_fail@{th}"] = int((g.t_oracle < th).sum())
            out[f"{grp}_det_ok_but_final_fail@{th}"] = int(((g.t_det >= th) & (g.t_final < th)).sum())
    return out


if __name__ == "__main__":
    part = sys.argv[1]
    tg = [("val" if a == "val" else int(a)) for a in sys.argv[2:]] or TARGETS
    res, allP = {}, []
    for t in tg:
        f = DUMP / tname(t) / f"D_{part}.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f)
        b = load_bundle(bundle_dir(t))["cfg"]["boundary"]
        if part == "train":
            ev = train_events(t)
        else:
            from srcv import core as C
            ev = C.real_events("val") if t == "val" else C.events(t, "test")
        P = per_event(D, ev, b)
        P["target"] = tname(t)
        allP.append(P)
        res[tname(t)] = table(P)
    P = pd.concat(allP, ignore_index=True)
    res["all"] = table(P)
    P.to_csv(HERE / f"diag_{part}_events.csv", index=False)
    (HERE / f"diag_{part}.json").write_text(json.dumps(res, indent=1))
    print(pd.DataFrame(res).T.to_string())

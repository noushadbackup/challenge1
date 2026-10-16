"""Boundary-model quality on training-OOF candidates: per detected evaluable GT fall, tIoU of replay median,
detector, model bounds, and the selected bounds of a few variants.   python3 mq.py <tag> [targets]"""
from common import *  # noqa
from diag import per_event
from select_b import cand_model, model_bounds, bounds
from uwbfall.pipeline import load_bundle

tag = sys.argv[1]
tg = [("val" if a == "val" else int(a)) for a in sys.argv[2:]] or TARGETS
rows = []
for t in tg:
    b = load_bundle(bundle_dir(t))["cfg"]["boundary"]
    D = pd.read_csv(DUMP / tname(t) / "D_train.csv")
    L = dict(np.load(BIG / "logits" / tag / tname(t) / "train.npz"))
    M = cand_model(D, L)
    P = per_event(D, train_events(t), b)
    P = P[P.detected.astype(bool)].copy()
    j = P.row.astype(int).values
    for how in ("argmax", "exp"):
        mb = np.array([model_bounds(M[i], b["dmin"], how) for i in j])
        P[f"t_model_{how}"] = tiou(mb[:, 0], mb[:, 1], P["gs"].values, P["ge"].values)
        P[f"es_model_{how}"] = mb[:, 0] - P["gs"].values
    for nm, v in {"selm1": {"m": 1.0, "tol": 0.1}, "selm0": {"m": 0.0, "tol": 0.1},
                  "selm0pm": {"m": 0.0, "tol": 0.1, "pool_model": True}}.items():
        a, e, _ = bounds(D, b, M, v)
        P[f"t_{nm}"] = tiou(a[j], e[j], P["gs"].values, P["ge"].values)
    P["target"] = tname(t)
    rows.append(P)
P = pd.concat(rows)
P["agree"] = P.agree.astype(bool)
cols = [c for c in P.columns if c.startswith("t_") and c not in ("t_crop_best",)]
for grp, g in (("all", P), ("agree", P[P.agree]), ("gate_fail", P[~P.agree])):
    print(f"--- {grp} n={len(g)}")
    print(pd.DataFrame({th: {c: round(float((g[c] >= th).mean()), 3) for c in cols} for th in (0.3, 0.5, 0.7)}).to_string())
print("median |start err| median-replay %.3f model-argmax %.3f model-exp %.3f" % (
    np.median(np.abs(P.err_s)), np.median(np.abs(P.es_model_argmax)), np.median(np.abs(P.es_model_exp))))

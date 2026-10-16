"""Train the TAL member for one SR-CV bundle: 4 inner-fold models on the bundle's training streams (the base
bundle's own GroupKFold stream folds) -> training OOF; 1 final model on all training streams -> test streams.

    python3 work/R3_tal/run.py <bundle> [--tag v1] [--steps 1000] [--h 48] [--win 512]
-> /home/user/data_raw/r3/tal/out/<tag>/<bundle>/{oof.npz, oof_segs.csv, test.npz, test_segs.csv, tal.pt, args.json}
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import tal

ap = argparse.ArgumentParser()
ap.add_argument("bundle")
ap.add_argument("--tag", default="v1")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--h", type=int, default=48)
ap.add_argument("--win", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--aux_w", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--inner", default="0,1,2,3")
a = ap.parse_args()
out = Path(f"/home/user/data_raw/r3/tal/out/{a.tag}/{a.bundle}")
out.mkdir(parents=True, exist_ok=True)
(out / "args.json").write_text(json.dumps(vars(a)))
tr, te = tal.load(a.bundle)
bseed = {"fold_0": 0, "fold_1": 1, "fold_2": 2, "fold_3": 3, "official_train": 4}[a.bundle] * 100 + a.seed * 1000
t0 = time.time()
def log(m): print(f"[{a.bundle} {time.time()-t0:6.0f}s] {m}", flush=True)
S = {sid: {"X": v["X"].astype(np.float32), "y": v["y"], "ev": v["ev"]} for sid, v in tr.items()}
fold_of = {sid: int(v["fold"][0]) for sid, v in tr.items()}

def outputs(model, items, dur):
    res, rows = {}, []
    for sid, X in items:
        segs, aux, pmax = tal.stream_outputs(model, X, dur[sid])
        res[f"aux__{sid}"], res[f"pmax__{sid}"] = aux, pmax
        res[f"cov__{sid}"] = tal.coverage(segs, len(X))
        rows += [(sid, s, e, p) for s, e, p in segs]
    return res, rows

if not (out / "oof.npz").exists():
    res, rows = {}, []
    for k in [int(x) for x in a.inner.split(",")]:
        trs = [S[s] for s in S if fold_of[s] != k]
        log(f"inner fold {k}: train on {len(trs)} streams")
        m = tal.train(trs, seed=bseed + k, steps=a.steps, win=a.win, batch=a.batch, h=a.h, aux_w=a.aux_w, log=log)
        r, rw = outputs(m, [(s, S[s]["X"]) for s in S if fold_of[s] == k], {s: tr[s]["dur"] for s in tr})
        res.update(r); rows += rw
    np.savez_compressed(out / "oof.npz", **res)
    pd.DataFrame(rows, columns=["stream_id", "t_start", "t_end", "score"]).to_csv(out / "oof_segs.csv", index=False)
    log("OOF done")
if not (out / "test.npz").exists():
    log("final model on all training streams")
    m = tal.train(list(S.values()), seed=bseed + 50, steps=a.steps, win=a.win, batch=a.batch, h=a.h, aux_w=a.aux_w, log=log)
    torch.save(m.state_dict(), out / "tal.pt")
    r, rw = outputs(m, [(s, v["X"].astype(np.float32)) for s, v in te.items()], {s: v["dur"] for s, v in te.items()})
    np.savez_compressed(out / "test.npz", **r)
    pd.DataFrame(rw, columns=["stream_id", "t_start", "t_end", "score"]).to_csv(out / "test_segs.csv", index=False)
    log("test done")

"""Fit the dense boundary model for one target: 4-fold OOF on the target's TRAINING streams (bundle stream folds)
plus a final model on all training streams applied to the target's test streams.

    python3 bfit.py <target> [--tag NAME] [--steps 400] [--no-p] [--seed 0]
Writes BIG/logits/<tag>/<target>/{train,test}.npz  ({sid: (2, n*10) float16 logits}).
"""
from common import *  # noqa
import argparse
import json
import time
import torch
from bmodel import stream_inputs, train_bnet, predict_logits
from uwbfall.pipeline import load_bundle


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def xcache(t, src, B, p, use_p, fine2=False):
    d = BIG / "x" / tname(t)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{src.sid}{'_p' if use_p else ''}{'_f2' if fine2 else ''}.npy"
    if f.exists():
        return np.load(f)
    X = stream_inputs(src, B, p, use_p, fine2)
    src.release()
    np.save(f, X.astype(np.float32))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-p", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--oof-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--all-falls", action="store_true", help="also train on flagged falls (bounds are annotator outputs)")
    ap.add_argument("--fine2", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    t = "val" if a.target == "val" else int(a.target)
    use_p = not a.no_p
    out = BIG / "logits" / a.tag / tname(t)
    out.mkdir(parents=True, exist_ok=True)
    B = load_bundle(bundle_dir(t))
    meta = B["meta"]
    z = np.load(bundle_dir(t) / "oof.npz")
    oofp = {s: z["oof_blend"][z["sid"] == s] for s in np.unique(z["sid"])}
    ev = train_events(t) if a.all_falls else unflagged(train_events(t))
    if a.all_falls and "ignore" in ev:
        ev = ev[~ev["ignore"].astype(str).str.lower().isin(["true", "1"])]
    srcs = sorted(train_sources(t), key=lambda s: s.sid)
    t0 = time.time()
    data = {}
    for s in srcs:
        X = xcache(t, s, B, oofp.get(s.sid), use_p, a.fine2)
        e = ev[ev.stream_id == s.sid][["t_start", "t_end"]].astype(float).values
        data[s.sid] = (X, e)
    log(f"[{tname(t)}] inputs {len(data)} streams, C={X.shape[1]}, {sum(len(v[1]) for v in data.values())} falls ({time.time()-t0:.0f} s)")
    kw = dict(steps=a.steps, sigma=a.sigma, h=a.h)
    # inner OOF on the training streams
    oof = {}
    for k, held in meta["folds"].items():
        tr = [data[s] for s in data if s not in held]
        ms = [train_bnet(tr, seed=a.seed + 10 * int(k) + 100 * j, log=log, **kw) for j in range(a.seeds)]
        for s in held:
            oof[s] = np.mean([predict_logits(m, data[s][0]) for m in ms], 0).astype(np.float16)
        log(f"[{tname(t)}] inner fold {k} done")
    np.savez_compressed(out / "train.npz", **oof)
    if a.oof_only:
        return
    ms = [train_bnet(list(data.values()), seed=a.seed + 1000 + 100 * j, log=log, **kw) for j in range(a.seeds)]
    for j, m in enumerate(ms):
        torch.save(m.state_dict(), out / f"final_{j}.pt")
    pt = np.load(DUMP / tname(t) / "p_test.npz")
    te = {}
    for s in sorted(test_sources(t), key=lambda s: s.sid):
        X = xcache(t, s, B, pt[s.sid], use_p, a.fine2)
        te[s.sid] = np.mean([predict_logits(m, X) for m in ms], 0).astype(np.float16)
    np.savez_compressed(out / "test.npz", **te)
    (out / "args.json").write_text(json.dumps(vars(a)))
    log(f"[{tname(t)}] done ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()

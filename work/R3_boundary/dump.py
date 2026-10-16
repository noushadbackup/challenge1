"""Dump replay internals (all 6 crops) for (a) the training-OOF candidates of each base bundle and (b) the test
candidates of each target, plus the per-stream blended detector probability. Checks that refine()+NMS reproduce
the bundle's oof_final.csv and the base fold/val CSV byte-for-byte (as DataFrames).

    python3 dump.py 0 [1 2 3 val]
"""
from common import *  # noqa
import time
import uwbfall
from uwbfall.pipeline import load_bundle, stream_probs, _nms, setup_torch
from uwbfall.extract import get_features
from uwbfall.boundary import replay_stream, refine, to_preds
from uwbfall.decode import to_events, events_df


def dump(t):
    out = DUMP / tname(t)
    out.mkdir(parents=True, exist_ok=True)
    B = load_bundle(bundle_dir(t))
    cfg = B["cfg"]
    _lim = setup_torch(cfg)  # noqa
    g = tuple(B["meta"]["postproc"])
    # ---- (a) training OOF candidates
    t0 = time.time()
    cand = pd.read_csv(bundle_dir(t) / "oof_candidates.csv")
    Ds = []
    for s in sorted(train_sources(t), key=lambda s: s.sid):
        c = cand[cand.stream_id == s.sid].reset_index(drop=True)
        if not len(c):
            continue
        F = get_features(s, cfg)
        Ds.append(replay_stream(s, F["prof_02"], c, cfg))
        s.release()
    D = pd.concat(Ds, ignore_index=True)
    D.to_csv(out / "D_train.csv", index=False)
    dur = {s.sid: s.duration for s in train_sources(t)}
    a, e, _ = refine(D, cfg["boundary"])
    fin = _nms(to_preds(D, a, e, dur), cfg, {})
    ref = pd.read_csv(bundle_dir(t) / "oof_final.csv")
    ok_tr = len(fin) == len(ref) and np.allclose(fin[["t_start", "t_end", "score"]].values, ref[["t_start", "t_end", "score"]].values)
    print(f"[{tname(t)}] train OOF: {len(D)} cands, reproduces oof_final: {ok_tr} ({time.time()-t0:.0f} s)", flush=True)
    # ---- (b) test candidates
    t0 = time.time()
    Ds, probs, rows = [], {}, []
    srcs = test_sources(t)
    dur = {s.sid: s.duration for s in srcs}
    for s in sorted(srcs, key=lambda s: s.sid):
        F = get_features(s, cfg)
        p, parts = stream_probs(B, F)
        probs[s.sid] = p.astype(np.float32)
        c = events_df(to_events(s.sid, p, *g))
        if len(c):
            Dk = replay_stream(s, F["prof_02"], c, cfg)
            Ds.append(Dk)
            a, e, _ = refine(Dk, cfg["boundary"])
            r = to_preds(Dk, a, e, dur)
            if len(r):
                rows.append(_nms(r, cfg, {}))
        s.release()
    D = pd.concat(Ds, ignore_index=True)
    D.to_csv(out / "D_test.csv", index=False)
    np.savez_compressed(out / "p_test.npz", **probs)
    pred = pd.concat(rows, ignore_index=True)
    pred.to_csv(out / "base_test.csv", index=False)
    ref = pd.read_csv(base_csv(t))
    ok_te = len(pred) == len(ref) and np.allclose(pred[["t_start", "t_end", "score"]].values, ref[["t_start", "t_end", "score"]].values)
    print(f"[{tname(t)}] test: {len(D)} cands, reproduces base CSV: {ok_te} ({time.time()-t0:.0f} s)", flush=True)
    return ok_tr, ok_te


if __name__ == "__main__":
    for a in sys.argv[1:]:
        dump("val" if a == "val" else int(a))

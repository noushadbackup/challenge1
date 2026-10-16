"""Dump the base TCN's standardised 112-channel block inputs (per bundle: its own PCA + tcn_norm) for every
stream that bundle sees (its training streams and its test streams), plus block labels / events / inner folds /
base OOF for the training streams.  One stream at a time, from the uwbfall feature cache.

    python3 work/R3_tal/dump.py fold_0 [fold_1 ... official_train]
-> /home/user/data_raw/r3/tal/feats/<bundle>.npz
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch
torch.set_num_threads(1)
import srcv
from srcv import core as C
import uwbfall
from uwbfall.config import load_config
from uwbfall.extract import get_features
from uwbfall.features import stream_blocks, design, block_labels
from uwbfall.models import standardise

RUNS = Path("/home/user/data_raw/srcv/runs/strict/norep")
OUT = Path("/home/user/data_raw/r3/tal/feats")


def parts(name):
    if name == "official_train":
        return (uwbfall.split_sources("train"), C.real_events("train"), uwbfall.split_sources("val"))
    k = int(name.split("_")[1])
    return srcv.pipeline_sources(k, "train"), C.events(k, "train"), srcv.pipeline_sources(k, "test")


def main(name):
    b = RUNS / name
    cfg = load_config(json.loads((b / "config.json").read_text()))
    pca = pickle.load(open(b / "pca.pkl", "rb"))
    z = np.load(b / "tcn_norm.npz"); mu, sd = z["mu"], z["sd"]
    oof = np.load(b / "oof.npz")
    tr, ev, te = parts(name)
    out = {}
    meta = {"train": [], "test": [], "events": {}}
    t0 = time.time()
    for part, srcs in (("train", tr), ("test", te)):
        for s in sorted(srcs, key=lambda s: s.sid):
            F = get_features(s, cfg)
            blk = stream_blocks(F, cfg)
            xp, xb, _ = design(blk, pca, cfg)
            X = standardise(np.hstack([xp.astype(np.float32), xb]), mu, sd, cfg["tcn"]["clip_z"])
            out[f"X__{s.sid}"] = X.astype(np.float16)
            meta[part].append(s.sid)
            if part == "train":
                y, fl = block_labels(ev, s.sid, blk["n"])
                m = oof["sid"] == s.sid
                assert m.sum() == blk["n"], (s.sid, m.sum(), blk["n"])
                out[f"y__{s.sid}"] = y.astype(np.int8)
                out[f"fold__{s.sid}"] = oof["fold"][m]
                for kk in ("oof_hgb", "oof_tcn", "oof_blend"):
                    out[f"{kk}__{s.sid}"] = oof[kk][m].astype(np.float32)
                e = ev[ev.stream_id == s.sid]
                fl = ((e.quality.astype(str) == "low") | (e.needs_review.astype(str).str.lower() == "true")).values
                out[f"ev__{s.sid}"] = np.c_[e.t_start.astype(float).values, e.t_end.astype(float).values, fl]
            meta.setdefault("dur", {})[s.sid] = float(s.duration)
            s.release()
            print(f"{name} {part} {s.sid} n={X.shape} {time.time()-t0:.0f}s", flush=True)
    out["meta"] = np.array(json.dumps(meta))
    np.savez(OUT / f"{name}.npz", **out)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)

"""Annotator boundaries for every interval any recovery rule can propose (union over each rule's grid).

For each proposed interval (one stream at a time, raw CIR of that stream only):
  * rep : the base system's replay stage (uwbfall.boundary.replay_stream + refine, bundle config: 6 crops on
          distinct frames, median of crops with tIoU-agreement >= 0.3, else detector bounds, 1.0 s floor);
  * seg : abc2027.annotate.propose_fall run once on the whole junction-to-junction segment holding the interval's
          centre (distinct frames, as the replay does), i.e. idea (a) "annotator on the segment crop".
Output: /home/user/data_raw/r3/recover/<target>/bounds_<part>.pkl  {(sid, t_start, t_end): {...}}

    python3 work/R3_recover/replay.py [target ...]
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

import srcv  # noqa: E402
import uwbfall  # noqa: E402
from uwbfall import pipeline as P  # noqa: E402
from uwbfall.boundary import propose_fall, refine, replay_stream, tracks_for  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from uwbfall.paths import RATE_HZ  # noqa: E402

import rules as RL  # noqa: E402

FS = RATE_HZ


def sources(target, part):
    if target == "official_train":
        return uwbfall.split_sources("train" if part == "train" else "val")
    return srcv.pipeline_sources(int(target.split("_")[1]), part)


def union(d) -> pd.DataFrame:
    U = [RL.propose(d, rule, prm, d["postproc"]) for rule in ("sec", "amax", "raw") for prm in RL.GRIDS[rule]]
    U += [RL.propose(d, "amax", prm, d["postproc"]) for prm in RL.GRIDS["segcrop"]]
    return pd.concat(U).drop_duplicates(["stream_id", "t_start", "t_end"]).reset_index(drop=True)


def run(target: str, part: str) -> None:
    out_p = RL.R / target / f"bounds_{part}.pkl"
    old = pickle.load(open(out_p, "rb")) if out_p.exists() else {}
    d = RL.load(target, part)
    cfg = P.load_bundle(f"/home/user/data_raw/srcv/runs/strict/norep/{target}")["cfg"]
    U = union(d)
    U = U[[(r.stream_id, r.t_start, r.t_end) not in old for r in U.itertuples()]]
    res = dict(old)
    for s in sources(target, part):
        c = U[U.stream_id == s.sid].reset_index(drop=True)
        if not len(c):
            continue
        F = get_features(s, cfg)
        D = replay_stream(s, F["prof_02"], c, cfg)
        a, e, use = refine(D, cfg["boundary"])
        segs = d["streams"][s.sid]["segs"]
        T = s.n_frames / FS
        seg_cache = {}
        for i, r in enumerate(c.itertuples()):
            m = 0.5 * (r.t_start + r.t_end) / RL.STEP
            j = min(int(np.searchsorted(segs[:, 1], m, side="right")), len(segs) - 1)
            if j not in seg_cache:
                ta, tb = segs[j, 0] * RL.STEP, segs[j, 1] * RL.STEP
                if tb - ta < 3.0:
                    cm = 0.5 * (ta + tb)
                    ta, tb = max(0.0, cm - 1.5), min(T, cm + 1.5)
                ia, ib = int(round(ta * FS)), int(round(tb * FS))
                pf = propose_fall(tracks_for(np.asarray(s.crop(ia, ib)), True), (ib - ia) / FS)
                seg_cache[j] = (ia / FS + pf.t_start, ia / FS + pf.t_end, pf.quality, ta, tb)
            ss, se, q, ta, tb = seg_cache[j]
            res[(r.stream_id, r.t_start, r.t_end)] = {"rep_s": float(a[i]), "rep_e": float(e[i]), "rep_use": bool(use[i]),
                                                       "seg_s": float(ss), "seg_e": float(se), "seg_q": q,
                                                       "seg_a": ta, "seg_b": tb}
        s.release()
        print(f"  {target} {part} {s.sid}: {len(c)} intervals", flush=True)
    with open(out_p, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["fold_0", "fold_1", "fold_2", "fold_3", "official_train"]:
        for part in ("train", "test"):
            t0 = time.time()
            run(t, part)
            print(f"{t} {part} done {time.time() - t0:.0f} s", flush=True)

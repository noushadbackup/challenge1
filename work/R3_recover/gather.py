"""R3 recover, step 1: dump the per-block quantities the recovery rules need, for every base (final_norep) bundle.

For each target (fold_0..3 and official_train):
  * train part (the bundle's own training streams): stream-grouped OOF blend (already gated by the subject-grouped
    OOF P(fall clip)), the raw blend, the block gate, the junction segments, and the base OOF final rows
    (oof_final.csv). The gate is recomputed with pipeline._fit_clip (same code/config as the bundle) and checked
    against oof.npz so that gate * raw reproduces the stored gated OOF exactly.
  * test part (fold test streams, or val for official_train): uwbfall.predict(..., return_debug=True) -> gated
    probability, gate, junction peaks, detector candidates; the final rows are checked byte-equal to the base CSV.

Nothing here is tuned; it only reads the frozen bundles. Output: /home/user/data_raw/r3/recover/<target>/{train,test}.pkl

    python3 work/R3_recover/gather.py [target ...]
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

import srcv  # noqa: E402
from srcv import core as C  # noqa: E402
import uwbfall  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall import pipeline as P  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from uwbfall.features import design, stream_blocks  # noqa: E402

BUNDLES = Path("/home/user/data_raw/srcv/runs/strict/norep")
BASE = Path("/home/user/challenge1/work/STRICT/results/norep")
OUT = Path("/home/user/data_raw/r3/recover")
STEP = 0.2


def sources_for(target: str, part: str):
    if target == "official_train":
        return (uwbfall.split_sources("train"), C.real_events("train")) if part == "train" else \
            (uwbfall.split_sources("val"), None)
    k = int(target.split("_")[1])
    return srcv.pipeline_sources(k, part), C.events(k, part)


def train_part(target: str) -> dict:
    b = BUNDLES / target
    B = P.load_bundle(b)
    cfg = B["cfg"]
    srcs, ev = sources_for(target, "train")
    srcs = sorted(srcs, key=lambda s: s.sid)
    z = np.load(b / "oof.npz")
    sid, y = z["sid"], z["y"]
    comps = {s.sid: CL.composition_of(s) for s in srcs}
    clip_in = []
    for s in srcs:
        F = get_features(s, cfg)
        blk = stream_blocks(F, cfg)
        xp, _xb, cols = design(blk, B["pca"], cfg)
        fpc = [i for i, c in enumerate(cols) if c.startswith("fp") and not c.startswith("fpseg")]
        clip_in.append((s, blk["base"], xp[:, fpc]))
        s.release()
    state, gate_oof = P._fit_clip(clip_in, comps, cfg, lambda m: print("  ", m, flush=True))
    assert abs(state["thr"] - B["clip"]["thr"]) < 1e-12
    raw = P._blend(cfg, z["oof_tcn"], z["oof_hgb"])
    out = {"streams": {}, "oof_final": pd.read_csv(b / "oof_final.csv"),
           "oof_candidates": pd.read_csv(b / "oof_candidates.csv"), "postproc": tuple(B["meta"]["postproc"]),
           "events": ev, "durations": {s.sid: s.duration for s in srcs}}
    for s, base, fp in clip_in:
        m = sid == s.sid
        g = gate_oof[s.sid]
        pg = CL.gate_probs(raw[m], g, cfg["clip"]["gate"], cfg["clip"]["gate_floor"])
        err = float(np.abs(pg - z["oof_blend"][m]).max())
        assert err < 1e-9, (s.sid, err)
        F = get_features(s, cfg)
        stat = CL.junction_stat(F["prof_02"], cfg["clip"]["junction_W"])
        pk = CL.junction_peaks(stat, state["thr"], cfg["clip"]["junction_min_sep_s"])
        segs = CL.segments_from_peaks(len(stat), pk)
        out["streams"][s.sid] = {"p": z["oof_blend"][m], "raw": raw[m], "gate": g, "segs": segs, "stat": stat,
                                 "y": y[m], "hgb": z["oof_hgb"][m], "tcn": z["oof_tcn"][m]}
    return out


def test_part(target: str) -> dict:
    b = BUNDLES / target
    srcs, ev = sources_for(target, "test")
    pred, cands, dbg = uwbfall.predict(b, srcs, return_debug=True)
    ref = BASE / ("val.csv" if target == "official_train" else f"fold_{target.split('_')[1]}.csv")
    tmp = OUT / target / "_check.csv"
    pred.to_csv(tmp, index=False)
    same = tmp.read_bytes() == ref.read_bytes()
    tmp.unlink()
    print(f"  {target} test rows byte-identical to base: {same}", flush=True)
    assert same
    streams = {}
    for s in srcs:
        d = dbg.get(s.sid)
        if d is None:
            continue
        n = len(d["p"])
        pk = np.round(np.asarray(d["peaks"]) / STEP).astype(int)
        streams[s.sid] = {"p": d["p"], "gate": d["gate"], "segs": CL.segments_from_peaks(n, pk),
                          "hgb": d["hgb"], "tcn": d["tcn"]}
    return {"streams": streams, "pred": pred, "cands": cands, "durations": {s.sid: s.duration for s in srcs}}


if __name__ == "__main__":
    for t in sys.argv[1:] or ["fold_0", "fold_1", "fold_2", "fold_3", "official_train"]:
        (OUT / t).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if not (OUT / t / "train.pkl").exists():
            with open(OUT / t / "train.pkl", "wb") as f:
                pickle.dump(train_part(t), f)
        print(f"{t} train done {time.time() - t0:.0f} s", flush=True)
        if not (OUT / t / "test.pkl").exists():
            with open(OUT / t / "test.pkl", "wb") as f:
                pickle.dump(test_part(t), f)
        print(f"{t} test done {time.time() - t0:.0f} s", flush=True)

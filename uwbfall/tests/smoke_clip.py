"""Integration check of the W3 clip stage inside uwbfall (fit helpers + predict) against the I1 experiment code.

1. NMS: the official_train bundle of SR-CV bag 0 with decode.one_event_per_clip="time" (gap 4 s) must give the
   same val CSV as work/I1_clip/preds/uwb_default/N4/val.csv (time NMS applied post hoc to the reference).
2. Gate: pipeline._fit_clip on the 13 train streams must reproduce the experiment's subject-grouped OOF gates
   (work/I1_clip/cache/prep_official.pkl), and predict with clip.gate="mult" (operating point copied from the
   experiment) must give work/I1_clip/preds/uwb_default/G/val.csv.

    python3 uwbfall/tests/smoke_clip.py
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import uwbfall
from uwbfall import clip as CL
from uwbfall import pipeline as PL
from uwbfall.config import load_config
from uwbfall.extract import get_features
from uwbfall.features import design, stream_blocks

REPO = Path(__file__).resolve().parents[2]
I1 = REPO / "work" / "I1_clip"
SRC = Path("/home/user/data_raw/srcv/runs/uwb_default/official_train")
TMP = Path("/home/user/data_raw/I1/smoke")
ok = True


def bundle(name, over, postproc=None, clip_state=None):
    d = TMP / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in ["meta.json", "pca.pkl", "hgb.pkl", "tcn_norm.npz", "tcn_0.pt", "tcn_1.pt", "tcn_2.pt"]:
        shutil.copy(SRC / f, d / f)
    cfg = json.loads((SRC / "config.json").read_text())
    cfg = load_config(cfg)
    for sec, kv in over.items():
        cfg[sec].update(kv)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    if postproc is not None:
        m = json.loads((d / "meta.json").read_text())
        m["postproc"] = postproc
        (d / "meta.json").write_text(json.dumps(m, indent=1))
    if clip_state is not None:
        with open(d / "clip.pkl", "wb") as f:
            pickle.dump(clip_state, f)
    return d


def compare(df, ref, label):
    global ok
    a = df.to_csv(index=False)
    b = pd.read_csv(ref).to_csv(index=False)
    same = a == b
    ok &= same
    print(f"{label}: {len(df)} rows, identical to {ref.relative_to(REPO)}: {same}", flush=True)


val = uwbfall.split_sources("val")
gate03 = {"gate": 0.3}
# 1. time NMS
b1 = bundle("nms_time", {"boundary": gate03, "decode": {"one_event_per_clip": "time", "nms_gap_s": 4.0}})
compare(PL.predict(b1, val), I1 / "preds" / "uwb_default" / "N4" / "val.csv", "predict time-NMS")

# 2. clip fit helpers + gate
cfg = load_config({**json.loads((SRC / "config.json").read_text()),
                   "clip": {"segclf": {"enabled": True}, "gate": "mult"}})
with open(SRC / "pca.pkl", "rb") as f:
    pca = pickle.load(f)
tr = sorted(uwbfall.split_sources("train"), key=lambda s: s.sid)
clip_in, comps = [], {}
for s in tr:
    F = get_features(s, cfg)
    blk = stream_blocks(F, cfg)
    xp, _xb, cols = design(blk, pca, cfg)
    fpc = [i for i, c in enumerate(cols) if c.startswith("fp") and not c.startswith("fpseg")]
    clip_in.append((s, blk["base"], xp[:, fpc]))
    comps[s.sid] = CL.composition_of(s)
state, gate_oof = PL._fit_clip(clip_in, comps, cfg, print)
P = pickle.load(open(I1 / "cache" / "prep_official.pkl", "rb"))
same_thr = abs(state["thr"] - P["thr"]) < 1e-12
same_gate = all(np.array_equal(gate_oof[k], P["gate"][k]) for k in gate_oof)
ok &= same_thr and same_gate
print(f"_fit_clip: threshold equal {same_thr}, OOF gates equal on {len(gate_oof)} train streams: {same_gate}")
ops = json.loads((I1 / "cache" / "ops_uwb_default_official.json").read_text())
b2 = bundle("gate_mult", {"boundary": gate03, "clip": {"segclf": {**cfg["clip"]["segclf"]}, "gate": "mult"}},
            postproc=ops["G"]["op"], clip_state=state)
compare(PL.predict(b2, val), I1 / "preds" / "uwb_default" / "G" / "val.csv", "predict gate=mult")
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

"""Detected junction peaks (deployable: fold bundle's clip threshold, features only) for every SR-CV test stream
and for val (official_train bundle). Saved to detj.json."""
import sys, json, time, pickle
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import numpy as np
import srcv
from uwbfall import clip as CL
from uwbfall.extract import get_features
from uwbfall.pipeline import load_bundle
from uwbfall.sources import parse_sources
out = {}
RUN = "/home/user/data_raw/srcv/runs/final_b0"
for k in list(range(4)) + ["official"]:
    bdir = f"{RUN}/fold_{k}" if k != "official" else f"{RUN}/official_train"
    cfg = json.load(open(f"{bdir}/config.json"))
    from uwbfall.config import load_config
    cfg = load_config(cfg)
    st = pickle.load(open(f"{bdir}/clip.pkl", "rb"))
    srcs = srcv.pipeline_sources(k, "test") if k != "official" else parse_sources("val")
    for s in srcs:
        t0 = time.time()
        F = get_features(s, cfg)
        stat = CL.junction_stat(F["prof_02"], cfg["clip"]["junction_W"])
        pk = CL.junction_peaks(stat, st["thr"], cfg["clip"]["junction_min_sep_s"]) * CL.STEP
        out[s.sid] = pk.tolist()
        s.release()
        print(k, s.sid, len(pk), round(time.time() - t0, 1), flush=True)
json.dump(out, open("detj.json", "w"))

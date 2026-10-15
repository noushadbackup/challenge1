"""Rule-compliance check after the organisers' answer to Q2 (no exploiting repeated content).

FINAL uses two things that compare frames within a stream:
  * clip.segclf features rep_rate / rep_sp2/3/4 / rep_n (sample-and-hold repeat statistics), and
  * boundary.dedup (the annotator replay drops bit-exact repeated frames).
This script measures STRICT variants that drop them, by re-tuning the existing FINAL bag-0 SR-CV bundles
(post-model stages only; HGB/TCN unchanged) and comparing with FINAL bag 0 under the promotion rule.

    python3 work/STRICT/run_strict.py [variant ...]      variants: norep, nodedup, strict (= both)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

import srcv
from srcv import core as C
import uwbfall
from uwbfall.pipeline import retune

HERE = Path(__file__).resolve().parent
RUNS = Path("/home/user/data_raw/srcv/runs/final_b0")
OUT = Path("/home/user/data_raw/srcv/runs/strict")
REF = Path(__file__).resolve().parents[2] / "srcv" / "results" / "final_b0"
VARIANTS = {
    "norep": {"clip": {"segclf": {"features": "phys"}}},
    "nodedup": {"boundary": {"dedup": False}},
    "strict": {"clip": {"segclf": {"features": "phys"}}, "boundary": {"dedup": False}},
}


def run(name: str) -> dict:
    ov = VARIANTS[name]
    res = HERE / "results" / name
    res.mkdir(parents=True, exist_ok=True)
    preds = {}
    for k in range(4):
        t0 = time.time()
        b = retune(RUNS / f"fold_{k}", srcv.pipeline_sources(k, "train"), C.events(k, "train"), ov,
                   OUT / name / f"fold_{k}")
        p = uwbfall.predict(b, srcv.pipeline_sources(k, "test"))
        p.to_csv(res / f"fold_{k}.csv", index=False)
        preds[k] = res / f"fold_{k}.csv"
        print(f"[{name}] fold {k} done ({time.time() - t0:.0f} s)", flush=True)
    b = retune(RUNS / "official_train", uwbfall.split_sources("train"), C.real_events("train"), ov,
               OUT / name / "official_train")
    uwbfall.predict(b, uwbfall.split_sources("val")).to_csv(res / "val.csv", index=False)
    ref = {k: REF / f"fold_{k}.csv" for k in range(4)}
    pooled = srcv.score_pooled(preds)
    pr = srcv.promote(ref, preds, val_A=REF / "val.csv", val_B=res / "val.csv")
    out = {"variant": name, "overrides": ov,
           "pooled_rank_f1": pooled["ranking_score"]["f1_mean"] if "ranking_score" in pooled else pooled,
           "promotion_vs_final_b0": pr}
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000], flush=True)
    return out


if __name__ == "__main__":
    for v in sys.argv[1:] or ["strict", "norep", "nodedup"]:
        run(v)

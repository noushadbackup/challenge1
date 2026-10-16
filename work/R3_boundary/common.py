"""Shared helpers for R3 'boundary' (sharper start/end boundaries).

Targets: fold 0..3 (SR-CV, subject-disjoint) and 'val' (13 official train streams -> 5 val streams).
For every target we use the BASE system final_norep bundles (no retraining of detector members):
  BASE_RUNS/fold_k            trained on cv{k}_train_* (13 virtual streams)
  BASE_RUNS/official_train    trained on the 13 official train streams
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

ROOT = Path("/home/user/challenge1")
HERE = ROOT / "work" / "R3_boundary"
BIG = Path("/home/user/data_raw/r3/boundary")
BASE_RUNS = Path("/home/user/data_raw/srcv/runs/strict/norep")
BASE_PRED = ROOT / "work" / "STRICT" / "results" / "norep"
TARGETS = [0, 1, 2, 3, "val"]


BAG = os.environ.get("R3_BAG", "")   # "" = base bag 0; "bag1"/"bag2" = composite norep bundles (bags.py)
DUMP = BIG / (f"dump_{BAG}" if BAG else "dump")


def bundle_dir(t):
    if BAG:
        return BIG / "bags" / BAG / ("official_train" if t == "val" else f"fold_{t}")
    return BASE_RUNS / ("official_train" if t == "val" else f"fold_{t}")


def train_sources(t):
    import srcv
    import uwbfall
    return uwbfall.split_sources("train") if t == "val" else srcv.pipeline_sources(t, "train")


def test_sources(t):
    import srcv
    import uwbfall
    return uwbfall.split_sources("val") if t == "val" else srcv.pipeline_sources(t, "test")


def train_events(t):
    from srcv import core as C
    return C.real_events("train") if t == "val" else C.events(t, "train")


def tname(t):
    return "val" if t == "val" else f"fold_{t}"


def base_csv(t):
    return BASE_PRED / ("val.csv" if t == "val" else f"fold_{t}.csv")


def unflagged(ev: pd.DataFrame) -> pd.DataFrame:
    """Evaluable GT (the official loader's ignore_flagged rule)."""
    e = ev.copy()
    for c in ("quality", "needs_review"):
        if c not in e:
            e[c] = ""
    tb = lambda c: e[c].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]) if c in e else False
    flag = (e["quality"].astype(str).str.strip().str.lower() == "low") | tb("needs_review") | tb("ignore")
    return e[~flag]


def tiou(a0, a1, b0, b1):
    a0, a1, b0, b1 = map(np.asarray, (a0, a1, b0, b1))
    i = np.maximum(0, np.minimum(a1, b1) - np.maximum(a0, b0))
    u = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(u > 0, i / np.where(u > 0, u, 1), 0.0)

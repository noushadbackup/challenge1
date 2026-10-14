"""Shared helpers for I2 (boundaries, D3 hedge, loss masking). Reuses the I1/R2_EVAL fitted bundles."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import uwbfall
from uwbfall import clip as CL
from uwbfall.config import load_config
from srcv import bridge, core as C

HERE = Path(__file__).resolve().parent
I1 = HERE.parent / "I1_clip"
R2 = HERE.parent / "R2_EVAL"
RUNS = C.DATA / "runs"
BAGS = ["uwb_default", "tcn_bag1", "tcn_bag2"]
TARGETS = ["0", "1", "2", "3", "official"]
REF_CFG = load_config(None)          # current reference (G7N4)
COLS = ["stream_id", "t_start", "t_end", "score", "t_alarm"]


def oname(t):
    return "val" if t == "official" else f"fold_{t}"


def tdir(t):
    return "official_train" if t == "official" else f"fold_{t}"


def target_sources(t):
    if t == "official":
        return uwbfall.split_sources("train"), C.real_events("train"), uwbfall.split_sources("val"), C.real_events("val")
    k = int(t)
    return (bridge.pipeline_sources(k, "train"), C.events(k, "train"), bridge.pipeline_sources(k, "test"),
            C.events(k, "test"))


def compositions(t):
    """(train composition, test composition) with fall_type / ignore / subject (SR-CV) or package columns."""
    if t == "official":
        comp = pd.read_csv(C.PKG / "streams_composition.csv")
        return comp[comp.stream_id.str.startswith("train")], comp[comp.stream_id.str.startswith("val")]
    k = int(t)
    return C.composition(k, "train"), C.composition(k, "test")


class Probs:
    """G7 gated blend probabilities of a bag on a target (train OOF and test), optionally with a replaced HGB
    member (dict with 'oof' array and '<sid>|hgb' test arrays)."""

    def __init__(self, bag, t, hgb=None, tcn=None):
        base, bagd = RUNS / "uwb_default" / tdir(t), RUNS / bag / tdir(t)
        self.cfg = load_config(json.loads((base / "config.json").read_text()))
        self.w = self.cfg["blend"]["w_tcn"]
        o0, o1 = np.load(base / "oof.npz"), np.load(bagd / "oof.npz")
        self.sid = o0["sid"]
        assert (self.sid == o1["sid"]).all()
        self.oof_hgb, self.oof_tcn = o0["oof_hgb"], o1["oof_tcn"]
        self.Pt = dict(np.load(R2 / "cache" / f"probs_{bag}_{t}.npz"))
        if hgb is not None:
            self.oof_hgb = hgb["oof"]
            self.Pt.update({k: v for k, v in hgb.items() if k.endswith("|hgb")})
        if tcn is not None:
            self.oof_tcn = tcn["oof"]
            self.Pt.update({k: v for k, v in tcn.items() if k.endswith("|tcn")})
        prep = pickle.load(open(I1 / "cache" / f"prep_{t}.pkl", "rb"))
        self.gates = prep["gate"]
        self.floor = REF_CFG["clip"]["gate_floor"]

    def oof(self, gated=True):
        p = self.w * self.oof_tcn + (1 - self.w) * self.oof_hgb
        if gated:
            g = np.concatenate([self.gates[x] for x in pd.unique(self.sid)])
            p = CL.gate_probs(p, g, "mult", self.floor)
        return {s: p[self.sid == s] for s in pd.unique(self.sid)}

    def test(self, s, gated=True):
        p = self.w * self.Pt[f"{s}|tcn"] + (1 - self.w) * self.Pt[f"{s}|hgb"]
        if gated:
            p = CL.gate_probs(p, self.gates[s], "mult", self.floor)
        return p


def ops_g7(bag, t):
    return json.loads((I1 / "cache" / f"ops_{bag}_{t}7.json").read_text())["G"]["op"]

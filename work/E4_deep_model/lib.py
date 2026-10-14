"""Shared helpers: data loading, prototype post-processing, official scoring."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/challenge1/analysis")
PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
sys.path.insert(0, str(PKG / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from abc2027 import evaluate as E  # noqa: E402

from prototype_detector import STEP, to_events, write_sub  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
GT = PKG / "streams_events.csv"
# identical grid to analysis/prototype_detector.py (thr, smooth, min_dur, gap)
GRID = [(thr, sm, md, gp) for thr in (0.3, 0.4, 0.5, 0.6, 0.7) for sm in (1, 3, 5) for md in (0.4, 0.8) for gp in (0.2, 0.6)]


@lru_cache(maxsize=1)
def load() -> dict:
    with np.load(CACHE / "blocks.npz") as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=2)
def durations(split: str) -> dict:
    out = {}
    for p in sorted(Path("/home/user/data_raw/feats").glob(f"{split}_*.npz")):
        with np.load(p) as z:
            out[p.stem] = float(z["e_raw"].shape[1]) / E.RATE_HZ
    return out


@lru_cache(maxsize=2)
def gt(ignore_flagged=True):
    return E.load_ground_truth(GT, ignore_flagged=ignore_flagged)


def events_from_probs(sid: np.ndarray, t: np.ndarray, p: np.ndarray, g) -> list:
    evs = []
    for s in pd.unique(sid):
        m = sid == s
        evs += to_events(s, t[m] - STEP / 2, p[m], *g)
    return evs


def score_inproc(evs: list, split: str) -> dict:
    """Official abc2027.evaluate code in-process (durations from the npz = h5 durations, verified by E1)."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        path = f.name
    write_sub(evs, path)
    preds = E.validate_predictions(Path(path), durations(split))
    Path(path).unlink()
    d = durations(split)
    return E.evaluate(gt(), preds, d, E.PRIMARY_THETA)


def tune(sid, t, p, split="train"):
    best = None
    for g in GRID:
        r = score_inproc(events_from_probs(sid, t, p, g), split)
        sc = r["ranking_score"]["f1_mean"]
        if best is None or sc > best[0]:
            best = (sc, g)
    return best


def official(evs: list, name: str, split: str = "val") -> dict:
    pred = OUT / f"{name}.csv"
    js = OUT / f"{name}.json"
    write_sub(evs, pred)
    cmd = [sys.executable, "-B", "-m", "abc2027.evaluate", "--gt", str(GT), "--pred", str(pred.resolve()),
           "--streams", str(PKG / "streams" / split), "--ignore-flagged", "--out", str(js.resolve())]
    subprocess.run(cmd, check=True, cwd=PKG, capture_output=True,
                   env={"PYTHONPATH": str(PKG / "src"), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    return json.loads(js.read_text())


def summary(r: dict) -> dict:
    bt = r["by_theta"]
    return {
        "rank_f1": round(r["ranking_score"]["f1_mean"], 4),
        "f1@0.1/0.3/0.5/0.7": [round(bt[k]["f1"], 3) for k in ("0.1", "0.3", "0.5", "0.7")],
        "tp_fp_fn@0.3": [bt["0.3"]["tp"], bt["0.3"]["fp"], bt["0.3"]["fn"]],
        "fa_per_h@0.3": round(bt["0.3"]["false_alarms_per_hour"], 2),
        "start_err_med@0.3": round(bt["0.3"]["start_error_median_abs"] or -1, 3),
        "mean_tiou@0.3": round(bt["0.3"]["mean_tiou"] or -1, 3),
    }

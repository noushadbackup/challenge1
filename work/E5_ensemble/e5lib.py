"""E5 shared helpers: data views, prototype post-processing, official scoring (in-process + CLI)."""
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
    """Snapshot of E4's block cache: the 64 prototype 'full' features (float64, bit-identical pipeline to
    analysis/prototype_detector.py) + 48 range-band channels, prototype GroupKFold(4)-by-stream folds."""
    with np.load(CACHE / "blocks_e4.npz") as z:
        d = {k: z[k] for k in z.files}
    d["proto_cols"] = [str(c) for c in d["proto_cols"]]
    d["band_cols"] = [str(c) for c in d["band_cols"]]
    return d


def lag_within(x: np.ndarray, sid: np.ndarray, k: int) -> np.ndarray:
    """x[t+k] within each stream, edge padded (as prototype shift())."""
    out = np.empty_like(x)
    for s in pd.unique(sid):
        idx = np.flatnonzero(sid == s)
        v = x[idx]
        y = np.roll(v, -k)
        if k > 0:
            y[-k:] = v[-1]
        elif k < 0:
            y[:-k] = v[0]
        out[idx] = y
    return out


def frame(split: str) -> pd.DataFrame:
    d = load()
    X = np.hstack([d[f"{split}_X_proto"], d[f"{split}_X_band"].astype(np.float64)])
    df = pd.DataFrame(X, columns=d["proto_cols"] + d["band_cols"])
    sid = d[f"{split}_sid"]
    # per-radar temporal context (the prototype only lags the max over radars)
    for r in range(3):
        for key in (f"static5_r{r}", f"static10_r{r}", f"diff_r{r}"):
            for k in (-10, -5, 5, 10):
                df[f"{key}@{k * STEP:+.0f}s"] = lag_within(df[key].values, sid, k)
    return df


STRUCT = ["jump_max", "t_since_jump", "t_until_jump", "seg_len", "seg_frac"]


def view_cols(df: pd.DataFrame, view: str) -> list[str]:
    d = load()
    proto, band = d["proto_cols"], d["band_cols"]
    if view == "proto":  # prototype 'full' set (64)
        return list(proto)
    if view == "fused":  # prototype full + range bands (112)
        return list(proto) + list(band)
    if view == "physics":  # hedge: no structure, no EMA (junction artefact), no fingerprint
        return [c for c in proto if c.startswith(("static", "diff"))] + [c for c in band] + ["jump_max"]
    if view.startswith("radar"):
        r = int(view[-1])
        own = [c for c in df.columns if (f"_r{r}" in c) and c.startswith(("static", "diff", "ema", "mot", "dprof"))]
        return own + STRUCT
    raise ValueError(view)


@lru_cache(maxsize=2)
def durations(split: str) -> dict:
    out = {}
    for p in sorted(Path("/home/user/data_raw/feats").glob(f"{split}_*.npz")):
        with np.load(p) as z:
            out[p.stem] = float(z["e_raw"].shape[1]) / E.RATE_HZ
    return out


@lru_cache(maxsize=1)
def gt():
    return E.load_ground_truth(GT, ignore_flagged=True)


def events_from_probs(sid: np.ndarray, t: np.ndarray, p: np.ndarray, g) -> list:
    evs = []
    for s in pd.unique(sid):
        m = sid == s
        evs += to_events(s, t[m] - STEP / 2, p[m], *g)
    return evs


def score_inproc(evs: list, split: str) -> dict:
    """Official abc2027.evaluate code in-process (durations from npz == h5 durations, verified by E1)."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        path = f.name
    write_sub(evs, path)
    preds = E.validate_predictions(Path(path), durations(split))
    Path(path).unlink()
    return E.evaluate(gt(), preds, durations(split), E.PRIMARY_THETA)


def rank(r: dict) -> float:
    return r["ranking_score"]["f1_mean"]


def tune(sid, t, p, split="train", grid=GRID):
    best = None
    for g in grid:
        sc = rank(score_inproc(events_from_probs(sid, t, p, g), split))
        if best is None or sc > best[0] + 1e-12:
            best = (sc, g)
    return best


def official(evs: list, name: str, split: str = "val") -> dict:
    """Official CLI (subprocess, h5 streams) — used for every reported val number."""
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
        "rank_f1": round(rank(r), 4),
        "f1@0.1/0.3/0.5/0.7": [round(bt[k]["f1"], 3) for k in ("0.1", "0.3", "0.5", "0.7")],
        "tp_fp_fn@0.3": [bt["0.3"]["tp"], bt["0.3"]["fp"], bt["0.3"]["fn"]],
        "fa_per_h@0.3": round(bt["0.3"]["false_alarms_per_hour"], 2),
        "start_err_med@0.3": round(bt["0.3"]["start_error_median_abs"] or -1, 3),
        "mean_tiou@0.3": round(bt["0.3"]["mean_tiou"] or -1, 3),
    }

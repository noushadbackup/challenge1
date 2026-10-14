"""Official evaluator wrappers (package code abc2027.evaluate, never re-implemented)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .paths import PKG
from abc2027 import evaluate as E  # noqa: E402  (paths puts the package src on sys.path)

COLS = ["stream_id", "t_start", "t_end", "score", "t_alarm"]


def gt_records(events: pd.DataFrame, ignore_flagged: bool = True) -> list[dict]:
    """Same records as E.load_ground_truth(csv, ignore_flagged) for an events table (written to a temp CSV
    and parsed by the official loader)."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        path = f.name
    events.to_csv(path, index=False)
    try:
        return E.load_ground_truth(Path(path), ignore_flagged=ignore_flagged)
    finally:
        Path(path).unlink()


def validate_df(df: pd.DataFrame, durations: dict[str, float], path: Path | None = None) -> list[dict]:
    """Write ``df`` as CSV (to ``path`` or a temp file) and run the official validate_predictions on it."""
    tmp = path is None
    if tmp:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            path = Path(f.name)
    df[[c for c in COLS if c in df.columns]].to_csv(path, index=False)
    try:
        return E.validate_predictions(Path(path), durations)
    finally:
        if tmp:
            Path(path).unlink()


def score_inproc(df: pd.DataFrame, gt: list[dict], durations: dict[str, float]) -> dict:
    preds = validate_df(df, durations) if len(df) else []
    return E.evaluate(gt, preds, durations, E.PRIMARY_THETA)


def official_cli(pred_csv: str | Path, split: str, out_json: str | Path | None = None,
                 gt_csv: str | Path | None = None) -> dict:
    """python -m abc2027.evaluate --gt ... --pred <abs> --streams streams/<split> --ignore-flagged --out <json>"""
    pred_csv = Path(pred_csv).resolve()
    out_json = Path(out_json).resolve() if out_json else pred_csv.with_suffix(".json")
    gt_csv = Path(gt_csv).resolve() if gt_csv else PKG / "streams_events.csv"
    cmd = [sys.executable, "-B", "-m", "abc2027.evaluate", "--gt", str(gt_csv), "--pred", str(pred_csv),
           "--streams", str(PKG / "streams" / split), "--ignore-flagged", "--out", str(out_json)]
    env = {"PYTHONPATH": str(PKG / "src"), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(cmd, cwd=PKG, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"official evaluator failed: {r.stderr.strip()[-2000:]}")
    rep = json.loads(out_json.read_text())
    rep["_cli_stdout"] = r.stdout.strip()
    return rep


def summary(r: dict) -> dict:
    bt = r["by_theta"]
    return {
        "rank_f1": round(r["ranking_score"]["f1_mean"], 4),
        "f1@0.1/0.3/0.5/0.7": [round(bt[k]["f1"], 3) for k in ("0.1", "0.3", "0.5", "0.7")],
        "tp_fp_fn@0.3": [bt["0.3"]["tp"], bt["0.3"]["fp"], bt["0.3"]["fn"]],
        "fa_per_h@0.3": round(bt["0.3"]["false_alarms_per_hour"], 2),
        "start_err_med@0.3": None if bt["0.3"]["start_error_median_abs"] is None else round(bt["0.3"]["start_error_median_abs"], 3),
        "mean_tiou@0.3": None if bt["0.3"]["mean_tiou"] is None else round(bt["0.3"]["mean_tiou"], 3),
    }

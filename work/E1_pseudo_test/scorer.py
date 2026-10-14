"""Official scoring without h5 files: abc2027.evaluate with stream durations read from feature npz.

Durations: T / 50 where T = e_raw.shape[1] (the extractor keeps every frame), i.e. exactly
cir.shape[1] / RATE_HZ as abc2027.evaluate.stream_durations computes from the h5.
Everything else (validation, matching, ignore zones, ranking) is the official code, imported.

CLI:  python3 scorer.py --gt events.csv --pred pred.csv --feats DIR --ids id1,id2|--prefix p1001_ --out report.json
Verify: python3 scorer.py --verify   (compares with the official CLI on the 5 real val streams)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
sys.path.insert(0, str(PKG / "src"))

import numpy as np  # noqa: E402
from abc2027 import evaluate as E  # noqa: E402

RATE_HZ = E.RATE_HZ


def durations_from_npz(folder: Path, ids: list[str]) -> dict[str, float]:
    out = {}
    for sid in ids:
        with np.load(Path(folder) / f"{sid}.npz") as z:
            out[sid] = float(z["e_raw"].shape[1]) / RATE_HZ
    return out


def score_csv(gt_csv: Path, pred_csv: Path, durations: dict[str, float], ignore_flagged: bool = True) -> dict:
    preds = E.validate_predictions(Path(pred_csv), durations)
    gt = E.load_ground_truth(Path(gt_csv), ignore_flagged=ignore_flagged)
    return E.evaluate(gt, preds, durations, E.PRIMARY_THETA)


def score_records(gt_csv: Path, preds: list[dict], durations: dict[str, float], ignore_flagged: bool = True) -> dict:
    """In-memory variant (records as validate_predictions returns them). Use score_csv for reported numbers."""
    gt = E.load_ground_truth(Path(gt_csv), ignore_flagged=ignore_flagged)
    return E.evaluate(gt, preds, durations, E.PRIMARY_THETA)


def official_cli(gt: Path, pred: Path, streams: Path, out: Path) -> dict:
    cmd = [sys.executable, "-B", "-m", "abc2027.evaluate", "--gt", str(gt), "--pred", str(Path(pred).resolve()),
           "--streams", str(streams), "--ignore-flagged", "--out", str(Path(out).resolve())]
    subprocess.run(cmd, check=True, cwd=PKG, capture_output=True,
                   env={"PYTHONPATH": str(PKG / "src"), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    return json.loads(Path(out).read_text())


def verify(tmp: Path) -> None:
    feats = Path("/home/user/data_raw/feats")
    ids = [f"val_{i:02d}" for i in range(5)]
    dur_npz = durations_from_npz(feats, ids)
    dur_h5 = E.stream_durations(PKG / "streams" / "val")
    print("durations equal (npz vs h5):", dur_npz == dur_h5, dur_npz)
    preds = [PKG / "sample_submission.csv"] + sorted(Path("/home/user/challenge1/reports/prototype").glob("val_*.csv"))
    tmp.mkdir(parents=True, exist_ok=True)
    for p in preds:
        mine = score_csv(PKG / "streams_events.csv", p, dur_npz)
        ref = official_cli(PKG / "streams_events.csv", p, PKG / "streams" / "val", tmp / f"cli_{p.stem}.json")
        same = json.dumps(mine, sort_keys=True) == json.dumps(ref, sort_keys=True)
        print(f"{p.name}: mine={mine['ranking_score']['f1_mean']:.6f} cli={ref['ranking_score']['f1_mean']:.6f} "
              f"full_report_identical={same}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--feats", type=Path, default=Path("/home/user/data_raw/pseudo/feats"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.verify:
        verify(Path(__file__).resolve().parent / "results" / "verify")
        return
    ids = a.ids.split(",") if a.ids else sorted(p.stem for p in a.feats.glob(f"{a.prefix}*.npz") if ".tmp" not in p.name)
    rep = score_csv(a.gt, a.pred, durations_from_npz(a.feats, ids))
    if a.out:
        a.out.write_text(json.dumps(rep, indent=2))
    print(f"ranking_f1_mean={rep['ranking_score']['f1_mean']:.4f}")


if __name__ == "__main__":
    main()

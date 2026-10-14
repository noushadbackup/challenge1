"""Integration check: E4 detections (TCN / TCN+prototype blend, both chosen on train) +
E2 boundary replay with its FROZEN config B (chosen on train OOF with prototype candidates).
Nothing is tuned here; val is scored once per candidate with the official CLI.
Usage: python3 combine_tcn_replay.py   (run from anywhere; ~10 min)
"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
E2 = HERE.parent / "E2_boundaries"
E4 = HERE.parent / "E4_deep_model" / "out"
sys.path.insert(0, str(E2))
import s3_select as S  # noqa: E402  (refine, to_preds, official_cli; frozen config in out/summary.json)

S.OUT = HERE / "out"
S.OUT.mkdir(exist_ok=True)
cfg = json.load(open(E2 / "out" / "summary.json"))["B_replay_post"]
res = {}
for tag, cand in (("tcn", E4 / "val_tcn_all.csv"), ("tcn75_proto25", E4 / "val_ens_tcn_all_w0.75.csv")):
    rep = S.OUT / f"replay_val_{tag}.csv"
    if not rep.exists():
        subprocess.run([sys.executable, "s2_replay.py", "val", str(cand), str(rep)], cwd=E2, check=True)
    D = pd.read_csv(rep)
    res[f"{tag}_detector_bounds"] = S.official_cli(S.to_preds(D, D.t_start.values, D.t_end.values, "val"), "val", f"val_{tag}_raw")
    s, e, use = S.refine(D, cfg)
    res[f"{tag}_+replayB"] = S.official_cli(S.to_preds(D, s, e, "val"), "val", f"val_{tag}_replayB")
    res[f"{tag}_+replayB"]["n_replaced"] = int(use.sum())
    print(tag, json.dumps({k: v for k, v in res.items() if k.startswith(tag)}), flush=True)
json.dump(res, open(S.OUT / "summary.json", "w"), indent=1)

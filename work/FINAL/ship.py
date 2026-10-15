"""Copy the runnable files of a bundle into the repo (models/<name>/) with a sha256 manifest.

    python3 work/FINAL/ship.py /home/user/data_raw/uwbfall_models/final final

Shipped = everything `uwbfall.pipeline.load_bundle` / `predict` needs (config, meta, pca, hgb, clip, TCN members,
normalisation, drift reference). Not shipped (training-only diagnostics, regenerable): oof.npz, oof_*.csv, train.log.
MANIFEST.sha256 is in `sha256sum -c` format and covers the files of the source bundle, so it verifies both copies.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

RUN = ["config.json", "meta.json", "pca.pkl", "hgb.pkl", "clip.pkl", "tcn_norm.npz", "tcn_0.pt", "tcn_1.pt",
       "tcn_2.pt", "drift_ref.json"]
src, name = Path(sys.argv[1]), sys.argv[2]
dst = Path(__file__).resolve().parents[2] / "models" / name
dst.mkdir(parents=True, exist_ok=True)
lines, files, total = [], {}, 0
for f in RUN:
    p = src / f
    if not p.exists():
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    shutil.copy2(p, dst / f)
    assert hashlib.sha256((dst / f).read_bytes()).hexdigest() == h
    lines.append(f"{h}  {f}")
    files[f] = {"sha256": h, "bytes": p.stat().st_size}
    total += p.stat().st_size
meta = json.loads((src / "meta.json").read_text())
(dst / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
(dst / "manifest.json").write_text(json.dumps({
    "bundle_source": str(src), "files": files, "total_bytes": total,
    "config_hash": meta.get("config_hash"), "git_rev_at_fit": meta.get("git_rev"),
    "train_streams": meta.get("train_streams"), "postproc_operating_point": meta.get("postproc"),
    "oof": meta.get("oof"), "env": meta.get("env"), "timings": meta.get("timings"),
    "not_shipped": ["oof.npz", "oof_candidates.csv", "oof_final.csv", "train.log"],
    "regenerate": "make train-final (python -m uwbfall train --config uwbfall/configs/final.json "
                  "--train-splits train,val --out <dir>)"}, indent=1, default=str))
print(f"shipped {len(files)} files, {total / 1e6:.2f} MB -> {dst}")

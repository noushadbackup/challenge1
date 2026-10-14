"""HGB-member variants for (b) D3 zone features and (c) ignore-clip zero weights, per target.

Uses uwbfall's own code (stream_blocks / design / hgb_design / ignore_clip_mask / fit_hgb) with the bundle's PCA
and CV folds, so that variant 'base' must reproduce the bundle's OOF HGB and the cached test HGB probabilities
(checked). HGB is deterministic -> one fit serves all 3 TCN bags.

    python3 work/I2_boundaries/hgbvar.py <targets> [variants: base,Z,W,ZW]
Output: cache/hgb_<V>_<t>.npz with oof (train blocks, bundle order) and '<sid>|hgb' test arrays.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, R2, RUNS, target_sources, tdir  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall.config import load_config  # noqa: E402
from uwbfall.extract import get_features  # noqa: E402
from uwbfall.features import block_labels, design, hgb_design, ignore_clip_mask, stream_blocks  # noqa: E402
from uwbfall.models import fit_hgb, setup_torch  # noqa: E402
from uwbfall.sources import events_for  # noqa: E402

ZONES = [[0, 40, 69], [1, 5, 22], [2, 5, 22]]
VARS = {"base": {}, "Z": {"features": {"zones": ZONES}}, "Z0": {"features": {"zones": [[0, 40, 69]]}}, "W": {"labels": {"ignore_clips": "zero_weight"}},
        "Nul": {"hgb": {"max_iter": 290}},  # near-null control: 3 % fewer boosting iterations
        "ZW": {"features": {"zones": ZONES}, "labels": {"ignore_clips": "zero_weight"}}}


def main():
    targets = sys.argv[1].split(",")
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARS)
    for t in targets:
        bd = RUNS / "uwb_default" / tdir(t)
        raw = json.loads((bd / "config.json").read_text())
        raw.pop("clip", None)
        z = np.load(bd / "oof.npz")
        sid_b, fold = z["sid"], z["fold"]
        pca = pickle.load(open(bd / "pca.pkl", "rb"))
        tr, evtr, te, _ = target_sources(t)
        tr = sorted(tr, key=lambda s: s.sid)
        ev = events_for(tr, evtr)
        for v in variants:
            t0 = time.time()
            cfg = load_config(raw)
            for sec, d in VARS[v].items():
                cfg[sec].update(d)
            _lim = setup_torch(cfg)  # noqa: F841
            X, Y, W = [], [], []
            for s in tr:
                b = stream_blocks(get_features(s, cfg), cfg)
                xp, _xb, _c = design(b, pca, cfg)
                X.append(hgb_design(b, xp, cfg))
                Y.append(block_labels(ev, s.sid, b["n"])[0].astype(np.int8))
                if cfg["labels"]["ignore_clips"] == "zero_weight":
                    W.append(np.where(ignore_clip_mask(ev, CL.composition_of(s), s.sid, b["n"]), 0.0, 1.0))
                s.release()
            X, y = np.vstack(X), np.concatenate(Y)
            w = np.concatenate(W) if W else None
            assert len(y) == len(sid_b)
            oof = np.zeros(len(y))
            for k in range(cfg["cv"]["n_folds"]):
                m = fit_hgb(X[fold != k], y[fold != k], cfg, None if w is None else w[fold != k])
                oof[fold == k] = m.predict_proba(X[fold == k])[:, 1]
            hgb = fit_hgb(X, y, cfg, w)
            out = {"oof": oof}
            for s in te:
                b = stream_blocks(get_features(s, cfg), cfg)
                xp, _xb, _c = design(b, pca, cfg)
                out[f"{s.sid}|hgb"] = hgb.predict_proba(hgb_design(b, xp, cfg))[:, 1]
                s.release()
            np.savez_compressed(HERE / "cache" / f"hgb_{v}_{t}.npz", **out)
            msg = f"{t} {v}: X {X.shape}, zero-weight blocks {0 if w is None else int((w == 0).sum())}"
            if v == "base":
                Pt = np.load(R2 / "cache" / f"probs_uwb_default_{t}.npz")
                d1 = np.abs(oof - z["oof_hgb"]).max()
                d2 = max(np.abs(out[k] - Pt[k]).max() for k in out if k != "oof")
                msg += f" | reproduces bundle OOF max|d|={d1:.2e}, test max|d|={d2:.2e}"
            print(msg + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

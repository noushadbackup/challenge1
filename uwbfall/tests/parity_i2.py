"""I2 integration checks (features.zones, labels.ignore_clips, boundary.agg/fallback flags).

1. fit() HGB-only on the 13 official train streams with features.zones = Z (or labels.ignore_clips = zero_weight)
   -> OOF HGB and val HGB probabilities equal work/I2_boundaries/cache/hgb_<V>_official.npz (the experiment).
2. Default config: hgb_design / seqs / fit_hgb paths unchanged (base OOF equals the uwb_default bundle's OOF).
3. TCN weighted loss: train_tcn with weights runs and a zero-weight-everywhere batch gives zero gradient.
4. boundary.agg='cluster' and boundary.fallback='contain' are no-ops on candidates with a single crop mode / an
   agreeing crop (refine equals the reference).

    python3 -m uwbfall.tests.parity_i2 [Z|W|base,...]   (default Z,W)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import uwbfall
from uwbfall import pipeline as PL
from uwbfall.boundary import refine
from uwbfall.config import load_config
from uwbfall.extract import get_features
from uwbfall.models import seqs, train_tcn
from srcv import core as C

I2 = Path(__file__).resolve().parents[2] / "work" / "I2_boundaries"
ZONES = [[0, 40, 69], [1, 5, 22], [2, 5, 22]]
OFF = {"tcn": {"enabled": False}, "clip": {"segclf": {"enabled": False}, "gate": "none"},
       "decode": {"one_event_per_clip": False}, "report": {"oof_replay": False}}
VARS = {"Z": {"features": {"zones": ZONES}}, "W": {"labels": {"ignore_clips": "zero_weight"}}, "base": {}}


def deep(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def check_fit(v):
    bundle = C.DATA / "runs" / "uwb_default" / "official_train"
    raw = json.loads((bundle / "config.json").read_text())
    raw.pop("clip", None)
    cfg = deep(deep(raw, OFF), VARS[v])
    with tempfile.TemporaryDirectory(dir="/home/user/data_raw") as d:
        out = PL.fit(uwbfall.split_sources("train"), C.real_events("train"), cfg, out_dir=d)
        oof = np.load(Path(out) / "oof.npz")["oof_hgb"]
        B = PL.load_bundle(out)
        val = {}
        for s in uwbfall.split_sources("val"):
            _p, parts = PL.stream_probs(B, get_features(s, B["cfg"]))
            val[s.sid] = parts["hgb"]
            s.release()
    if v == "base":
        ref = np.load(bundle / "oof.npz")["oof_hgb"]
        okv = True
    else:
        z = np.load(I2 / "cache" / f"hgb_{v}_official.npz")
        ref = z["oof"]
        okv = all(np.array_equal(val[k], z[f"{k}|hgb"]) for k in val)
    ok = np.array_equal(oof, ref) and okv
    print(f"[{v}] fit() OOF HGB identical: {np.array_equal(oof, ref)}; val HGB identical: {okv} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_tcn_weights():
    """Unit weights reproduce the unweighted BCE path up to float rounding (sum/sum vs mean); all-zero weights give
    zero loss and zero gradient (an ignore-zone window teaches nothing)."""
    from uwbfall.models import _loss, _wloss, build_tcn
    cfg = load_config({"tcn": {"epochs": 1, "steps": 3, "win": 50, "batch": 2}})
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = (rng.random(400) < 0.2).astype(np.int8)
    sid = np.repeat(["a", "b"], 200)
    m1 = train_tcn(seqs(X, y, sid, np.ones(400, bool)), cfg, seed=1, epochs=1)
    m2 = train_tcn(seqs(X, y, sid, np.ones(400, bool), np.ones(400)), cfg, seed=1, epochs=1)
    d = max(float((a - b).abs().max()) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    torch.manual_seed(0)
    net = build_tcn(8, cfg).eval()  # no dropout: the two losses see the same outputs
    xb = torch.randn(2, 8, 50)
    yb = (torch.rand(2, 50) < 0.2).float()
    l1, l2 = _loss(net(xb), yb), _wloss(net(xb), yb, torch.ones(2, 50))
    net.zero_grad()
    l0 = _wloss(net(xb), yb, torch.zeros(2, 50))
    l0.backward()
    g0 = max(float(p.grad.abs().max()) for p in net.parameters() if p.grad is not None)
    ok = d < 1e-4 and abs(float(l1.detach()) - float(l2.detach())) < 1e-6 and float(l0.detach()) == 0.0 and g0 == 0.0
    print(f"[tcn] unit-weight vs unweighted: loss diff {abs(float(l1.detach()) - float(l2.detach())):.1e}, trained-weight max diff {d:.1e}; "
          f"zero weights: loss {float(l0.detach())}, max |grad| {g0} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_boundary():
    D = pd.DataFrame({"stream_id": ["x"], "t_start": [10.0], "t_end": [12.0], "score": [0.9]})
    for c in ["loc2_3d", "loc2_5d", "det2_3d", "det2_5d", "detq2_3d", "detq2_5d"]:
        D[f"{c}_s"], D[f"{c}_e"] = 10.2, 11.9
    b = load_config(None)["boundary"]
    r = [refine(D, {**b, **o}) for o in ({}, {"agg": "cluster"}, {"fallback": "contain"})]
    ok = all(np.allclose(x[0], r[0][0]) and np.allclose(x[1], r[0][1]) for x in r)
    print(f"[boundary] cluster / contain equal the median on a single-mode candidate: {ok}")
    return ok


if __name__ == "__main__":
    vs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Z", "W"]
    res = [check_tcn_weights(), check_boundary()] + [check_fit(v) for v in vs]
    print("ALL OK" if all(res) else "SOME FAILED")

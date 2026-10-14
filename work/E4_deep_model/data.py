"""Build the per-0.2 s block feature sequences used by the deep model (and the HGB reference).

Reuses analysis/prototype_detector.py (same block features, same train-only PCA fingerprint)
and adds per-radar range-band channels from the cached mot_map / prof_02.
Output: cache/blocks.npz  (all 18 streams concatenated, with sid / t / y / flagged / fold).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/challenge1/analysis")
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from common import FEATS, frame_labels, stream_ids  # noqa: E402
from prototype_detector import STEP, add_fingerprint, build, cols_for  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
# range bands (bin index ranges); mats: R29444/R30722 ~5-21, R29171 ~40-68 (data verdict F5)
BANDS = [(0, 5), (5, 13), (13, 22), (22, 40), (40, 53), (53, 69), (69, 100), (100, 184)]


def band_features(sid: str, n: int) -> tuple[np.ndarray, list[str]]:
    with np.load(FEATS / f"{sid}.npz") as z:
        mm = z["mot_map"][:, :n].astype(np.float32)  # (3, n, 184) EMA residual magnitude
        p = z["prof_02"][:, :n].astype(np.float32) / 1e3
    zc = p[..., :184] + 1j * p[..., 184:]
    # centred block-to-block profile change per bin (static change / junction / motion)
    dz = np.abs(np.roll(zc, -1, axis=1) - np.roll(zc, 1, axis=1))
    dz[:, 0], dz[:, -1] = dz[:, 1], dz[:, -2]
    cols, names = [], []
    for nm, arr in (("mot", mm), ("dprof", dz)):
        for r in range(3):
            for a, b in BANDS:
                v = np.log10(arr[r, :, a:b].mean(1) + 1e-9)
                cols.append(v - np.median(v))  # per-stream normalisation
                names.append(f"{nm}_r{r}_b{a}")
    return np.stack(cols, 1).astype(np.float32), names


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    sids = stream_ids()
    tr_ids = [s for s in sids if s.startswith("train")]
    va_ids = [s for s in sids if s.startswith("val")]
    tr, pca = add_fingerprint(build(tr_ids), None)
    va, _ = add_fingerprint(build(va_ids), pca)
    proto_cols = cols_for(tr, "full")
    out = {}
    for name, df in (("train", tr), ("val", va)):
        bands, bnames = [], None
        flagged = []
        for sid, d in df.groupby("sid", sort=False):
            b, bnames = band_features(sid, len(d))
            bands.append(b)
            flagged.append(frame_labels(sid, len(d), STEP)["flagged"])
        out[name] = dict(
            X_proto=df[proto_cols].values.astype(np.float64),
            X_band=np.vstack(bands),
            sid=np.asarray(df.sid.tolist(), dtype="U16"),
            t=df.t.values.astype(np.float64),
            y=df.y.values.astype(np.int8),
            flagged=np.concatenate(flagged).astype(bool),
        )
    # same leave-streams-out folds as the prototype (GroupKFold(4) by stream on train blocks)
    fold = np.full(len(tr), -1, np.int8)
    for k, (_a, b) in enumerate(GroupKFold(4).split(tr, tr.y, tr.sid)):
        fold[b] = k
    np.savez_compressed(
        CACHE / "blocks.npz",
        proto_cols=np.array(proto_cols), band_cols=np.array(bnames), train_fold=fold,
        **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()},
    )
    for k in range(4):
        print("fold", k, sorted(set(tr.sid[fold == k])))
    print("train", out["train"]["X_proto"].shape, out["train"]["X_band"].shape, "val", out["val"]["X_proto"].shape)


if __name__ == "__main__":
    main()

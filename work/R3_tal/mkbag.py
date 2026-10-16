"""Bag several TAL seeds: per-block members averaged (aux, pmax); segments = soft-NMS of the concatenated
per-seed segments; cov recomputed from the bagged segments. Same rule at predict time (fuse.patched_stream_probs).

    python3 work/R3_tal/mkbag.py <bag_tag> <tag1> <tag2> ... [--bundles fold_0,...]
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
import tal

OUT = Path("/home/user/data_raw/r3/tal/out")
args = sys.argv[1:]
bundles = ["fold_0", "fold_1", "fold_2", "fold_3", "official_train"]
if "--bundles" in args:
    i = args.index("--bundles"); bundles = args[i + 1].split(","); args = args[:i] + args[i + 2:]
bag, tags = args[0], args[1:]


def merge(npzs, segcsvs, n_of):
    res, rows = {}, []
    segs_all = [pd.read_csv(c) for c in segcsvs]
    for sid, n in n_of.items():
        for k in ("aux", "pmax"):
            res[f"{k}__{sid}"] = np.mean([z[f"{k}__{sid}"] for z in npzs], 0)
        S = np.vstack([d[d.stream_id == sid][["t_start", "t_end", "score"]].values for d in segs_all])
        S = tal.soft_nms(S) if len(S) else np.zeros((0, 3))
        res[f"cov__{sid}"] = tal.coverage(S, n)
        rows += [(sid, a, b, s) for a, b, s in S]
    return res, pd.DataFrame(rows, columns=["stream_id", "t_start", "t_end", "score"])


for b in bundles:
    dst = OUT / bag / b
    dst.mkdir(parents=True, exist_ok=True)
    srcs = [OUT / t / b for t in tags]
    for part, seg in (("oof", "oof_segs"), ("test", "test_segs")):
        zs = [np.load(s / f"{part}.npz") for s in srcs]
        n_of = {k.split("__", 1)[1]: len(zs[0][k]) for k in zs[0].files if k.startswith("aux__")}
        r, d = merge(zs, [s / f"{seg}.csv" for s in srcs], n_of)
        np.savez_compressed(dst / f"{part}.npz", **r)
        d.to_csv(dst / f"{seg}.csv", index=False)
    for i, s in enumerate(srcs):
        shutil.copy(s / "tal.pt", dst / f"tal_{i}.pt")
    a = json.loads((srcs[0] / "args.json").read_text()); a["bag"] = tags
    (dst / "args.json").write_text(json.dumps(a))
    print(b, "bagged", tags, flush=True)

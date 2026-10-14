"""Build SR-CV: choose the 4 subject-disjoint folds (folds.json) and write each fold's regenerated plans.

    python3 -m srcv.build            # folds.json (if missing) + /home/user/data_raw/srcv/fold_k/{train,test}_*.csv
    python3 -m srcv.build --refold   # re-run the fold search (changes every downstream number!)

Fold search (deterministic, rng 2027): every subject is held out exactly once; each fold holds 2 D1 + 2 D2 +
2 D3 fall subjects + 2 ADL subjects. Objective (lexicographic-ish weighted sum): balance evaluable test falls per
fold (P3_3 is the only D3 subject with evaluable falls, so its fold gets fewer evaluable D1/D2), balance D1 and D2
evaluable counts separately, at most 2 original-val fall subjects per fold, the two val ADL subjects in different
folds, and balanced ADL seconds.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import core as C

SEEDS = {"train": 7100, "test": 7200}  # + fold index


def search_folds(n_iter: int = 200_000, seed: int = 2027) -> list[dict]:
    t = C.subjects_table()
    groups = {d: t[t.dataset == d].reset_index(drop=True) for d in ("D1", "D2", "D3", "ADL")}
    rng = np.random.default_rng(seed)
    ne = {d: g.n_eval.to_numpy(float) for d, g in groups.items()}
    isval = {d: (g.origin == "val").to_numpy(float) for d, g in groups.items()}
    adls = groups["ADL"].adl_s.to_numpy(float)
    best = None
    for _ in range(n_iter):
        assign = {d: rng.permutation(len(g)) % 4 for d, g in groups.items()}  # 8 items -> 2 per fold
        ev = {d: np.bincount(assign[d], weights=ne[d], minlength=4) for d in ("D1", "D2", "D3")}
        tot = ev["D1"] + ev["D2"] + ev["D3"]
        nval = sum(np.bincount(assign[d], weights=isval[d], minlength=4)
                   for d in ("D1", "D2", "D3"))
        adl_val = np.bincount(assign["ADL"], weights=isval["ADL"], minlength=4)
        adl_s = np.bincount(assign["ADL"], weights=adls, minlength=4)
        if nval.max() > 2 or adl_val.max() > 1:
            continue
        cost = tot.var() + 0.5 * (ev["D1"].var() + ev["D2"].var()) + 1e-5 * adl_s.var()
        if best is None or cost < best[0] - 1e-12:
            best = (cost, {d: a.copy() for d, a in assign.items()})
    _cost, assign = best
    out = []
    for k in range(4):
        ts = {d: sorted(groups[d].subject[assign[d] == k].tolist()) for d in ("D1", "D2", "D3", "ADL")}
        held = [s for v in ts.values() for s in v]
        sel = t[t.subject.isin(held)]
        out.append({"fold": k, "test_subjects": ts,
                    "seeds": {"train": SEEDS["train"] + k, "test": SEEDS["test"] + k},
                    "test_n_falls": int(sel.n_falls.sum()), "test_n_evaluable": int(sel.n_eval.sum()),
                    "test_adl_s": round(float(sel.adl_s.sum()), 2)})
    return out


def main() -> None:
    if "--refold" in sys.argv or not C.FOLDS_JSON.exists():
        fl = search_folds()
        C.FOLDS_JSON.write_text(json.dumps({"density": C.DENSITY, "t_last_mode": C.T_LAST_MODE, "folds": fl}, indent=1))
        print("folds.json written")
    doc = json.loads(C.FOLDS_JSON.read_text())
    for f in doc["folds"]:
        k = f["fold"]
        print(f"fold {k}: test subjects {f['test_subjects']} (evaluable falls {f['test_n_evaluable']})")
        f["generated"] = {}
        for part in C.PARTS:
            info = C.generate(k, part)
            st = info["streams"]
            hours = sum(s["n_frames"] for s in st) / 50 / 3600
            nev = sum(s["n_events"] for s in st)
            f["generated"][part] = {"n_streams": len(st), "hours": round(hours, 4), "n_events": nev,
                                    "n_evaluable_test_rules": sum(s["n_evaluable"] for s in st),
                                    "falls_per_hour": round(nev / hours, 3),
                                    "composition_sha256": info["composition_sha256"],
                                    "streams": st}
            print(f"  {part}: {len(st)} streams, {hours:.2f} h, {nev} events "
                  f"({f['generated'][part]['n_evaluable_test_rules']} evaluable), {nev / hours:.2f} falls/h")
    C.FOLDS_JSON.write_text(json.dumps(doc, indent=1))
    (C.DATA / "folds.json").write_text(json.dumps(doc, indent=1))
    C._comp.cache_clear()


if __name__ == "__main__":
    main()

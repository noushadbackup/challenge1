"""Run the uwbfall pipeline on SR-CV: fit on each fold's regenerated train streams, predict its test streams.

    python3 -m srcv.run_pipeline <name> [--config cfg.json] [--folds 0,1,2,3]
    python3 -m srcv.run_pipeline <name> [--config cfg.json] --val-only   # official train -> real val (rule clause c)

Writes srcv/results/<name>/fold_k.csv (predictions) and summary.json (per-fold + pooled official scores and a
subject-cluster bootstrap CI). Model bundles go to /home/user/data_raw/srcv/runs/<name>/fold_k (outside git).
Features are taken from srcv's cache (copied into uwbfall's cache by bridge.seed_uwbfall_cache).
Compare two runs with:  python3 -m srcv.compare <nameA> <nameB> [--val-a X --val-b Y]
"""
from __future__ import annotations

import json
import resource
import sys
import time

from . import bridge
from . import core as C
from . import scoring as S


def main() -> None:
    a = sys.argv[1:]
    name = a[0]
    cfg = None
    if "--config" in a:
        cfg = json.loads(open(a[a.index("--config") + 1]).read())
    folds = [int(x) for x in a[a.index("--folds") + 1].split(",")] if "--folds" in a else [0, 1, 2, 3]
    import uwbfall

    out = C.HERE / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    if "--val-only" in a:
        t0 = time.time()
        bundle = uwbfall.fit(uwbfall.split_sources("train"), C.real_events("train"), cfg,
                             out_dir=C.DATA / "runs" / name / "official_train")
        pred = uwbfall.predict(bundle, uwbfall.split_sources("val"))
        pred.to_csv(out / "val.csv", index=False)
        rep = S.summary(S.score_val(out / "val.csv"))
        (out / "val_summary.json").write_text(json.dumps({"val": rep, "sec": round(time.time() - t0, 1)}, indent=1,
                                                          default=float))
        print(f"[{name}] official val rank F1 {rep['rank_f1']:.4f} tp/fp/fn@0.3 {rep['tp@0.3']}/{rep['fp@0.3']}/"
              f"{rep['fn@0.3']}", flush=True)
        return
    res = {"name": name, "config": cfg, "per_fold": []}
    t_all = time.time()
    for k in folds:
        t0 = time.time()
        for part in C.PARTS:
            bridge.seed_uwbfall_cache(k, part, uwbfall.load_config(cfg))
        bundle = uwbfall.fit(bridge.pipeline_sources(k, "train"), C.events(k, "train"), cfg,
                             out_dir=C.DATA / "runs" / name / f"fold_{k}")
        pred = uwbfall.predict(bundle, bridge.pipeline_sources(k, "test"))
        pred.to_csv(out / f"fold_{k}.csv", index=False)
        rep = S.summary(S.score(out / f"fold_{k}.csv", k))
        res["per_fold"].append({"fold": k, "test": rep, "sec": round(time.time() - t0, 1)})
        print(f"[{name}] fold {k}: rank F1 {rep['rank_f1']:.4f} tp/fp/fn@0.3 {rep['tp@0.3']}/{rep['fp@0.3']}/"
              f"{rep['fn@0.3']} ({time.time() - t0:.0f}s)", flush=True)
    if len(folds) == 4:
        preds = {k: out / f"fold_{k}.csv" for k in range(4)}
        res["pooled"] = S.summary(S.score_pooled(preds))
        res["bootstrap"] = S.bootstrap_ci(preds)
        print(f"[{name}] POOLED rank F1 {res['pooled']['rank_f1']:.4f} CI {res['bootstrap']['ci95']}")
    res["wall_s"] = round(time.time() - t_all, 1)
    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

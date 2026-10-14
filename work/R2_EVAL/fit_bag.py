"""Fit one extra TCN-only bag (3 members, re-seeded by tcn.seed_offset) on every SR-CV fold-train and on the
13 official train streams. HGB is not refitted: it is deterministic and is taken from the uwb_default bundles.

    python3 work/R2_EVAL/fit_bag.py <bag_name> <seed_offset> [targets: 0,1,2,3,official]
Bundles -> /home/user/data_raw/srcv/runs/<bag_name>/{fold_k,official_train}
"""
import sys, time
import srcv
from srcv import core as C, bridge
import uwbfall

name, off = sys.argv[1], int(sys.argv[2])
targets = sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1", "2", "3", "official"]
cfg = {"hgb": {"enabled": False}, "tcn": {"seed_offset": off}, "report": {"oof_replay": False}}
for t in targets:
    t0 = time.time()
    if t == "official":
        out = C.DATA / "runs" / name / "official_train"
        if (out / "meta.json").exists():
            continue
        uwbfall.fit(uwbfall.split_sources("train"), C.real_events("train"), cfg, out_dir=out)
    else:
        k = int(t)
        out = C.DATA / "runs" / name / f"fold_{k}"
        if (out / "meta.json").exists():
            continue
        bridge.seed_uwbfall_cache(k, "train", uwbfall.load_config(cfg))
        uwbfall.fit(bridge.pipeline_sources(k, "train"), C.events(k, "train"), cfg, out_dir=out)
    print(f"[{name}] {t} done in {time.time() - t0:.0f}s", flush=True)

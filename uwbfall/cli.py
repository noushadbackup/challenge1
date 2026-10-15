"""Command line interface.

  python -m uwbfall train   --config cfg.json --train-splits train[,val] --out <model_dir>
  python -m uwbfall predict --models <model_dir> --split val|test --out <csv>
  python -m uwbfall retune  --models <model_dir> --config cfg.json --train-splits train --out <new_model_dir>
                            (post-model stages only: clip stage, operating point, NMS; no HGB/TCN refit)
  python -m uwbfall hitattach --models <model_dir> --hit <hit.pkl> [--tau 0.46|none] [--nms-order detector]
                            --out <new_model_dir>
                            (W5 event hit model as score column + threshold; not promoted, see reports/round2/I3.md)
  python -m uwbfall driftref --models <model_dir> --train-splits train[,val]   (drift-monitor reference)
  python -m uwbfall score   --pred <csv> --split val [--out <json>]
  python -m uwbfall extract --sources train,val,test          (warm the feature cache)

Source specs (--train-splits / --sources): comma-separated items, each a package split name
(train | val | test), dir:<folder>[:<prefix>] or comp:<composition.csv>[:<events.csv>] (virtual streams).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


def _cmd_train(a) -> int:
    from .pipeline import fit
    from .paths import PKG
    from .sources import parse_sources

    srcs = parse_sources(a.train_splits)
    ev = pd.read_csv(a.events or PKG / "streams_events.csv", dtype=str)
    t0 = time.time()
    fit(srcs, ev, a.config, a.out)
    print(json.dumps({"model_dir": str(Path(a.out).resolve()), "wall_s": round(time.time() - t0, 1)}))
    return 0


def _cmd_predict(a) -> int:
    from abc2027 import evaluate as E

    from .paths import PKG
    from .pipeline import load_bundle, predict
    from .scoring import validate_df
    from .sources import parse_sources

    if not (a.split or a.sources):
        raise SystemExit("predict: give --split or --sources")
    spec = a.sources or a.split
    srcs = parse_sources(spec)
    t0 = time.time()
    B = load_bundle(a.models)
    if a.boundary:  # predict-time override of the boundary stage only (it does not depend on training)
        from .config import load_config
        over = json.loads(Path(a.boundary).read_text()) if Path(a.boundary).exists() else json.loads(a.boundary)
        over = over.get("boundary", over)
        B["cfg"] = load_config({**{k: v for k, v in B["cfg"].items() if k in load_config(None)},
                                "boundary": {**B["cfg"]["boundary"], **over}})
    if a.nms is not None:  # predict-time override of the final hard 1-D NMS (post-replay, training-independent)
        mode = {"off": False, "time": "time"}[a.nms]
        B["cfg"]["decode"] = {**B["cfg"]["decode"], "one_event_per_clip": mode,
                              **({"nms_gap_s": a.nms_gap} if a.nms_gap is not None else {})}
    mon = {}
    pred, cand, _ = predict(B, srcs, return_debug=True, monitor=mon)
    # official validation against the official durations (stream_durations for package splits)
    durations = {s.sid: s.duration for s in srcs}
    if a.split and not a.sources:
        durations = E.stream_durations(PKG / "streams" / a.split)
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    recs = validate_df(pred, durations, tmp)  # raises E.ValidationError on any invalid row
    tmp.replace(out)
    if a.candidates:
        cand.to_csv(a.candidates, index=False)
    # label-free drift monitors vs the bundle's training streams (uwbfall/drift.py)
    from . import drift
    rep = drift.report(B.get("drift_ref"), mon, pred)
    print(drift.format_report(rep), flush=True)
    dj = out.with_name(out.stem + "_drift.json")
    dj.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"pred": str(out), "rows": len(recs), "streams": len(srcs), "wall_s": round(time.time() - t0, 1),
                      "drift_alerts": rep["n_alert"], "recommend_fallback": rep["recommend_fallback"],
                      "drift_json": str(dj)}))
    return 0


def _cmd_driftref(a) -> int:
    from .config import load_config
    from .drift import reference_for_bundle
    from .sources import parse_sources

    cfg = load_config(json.loads((Path(a.models) / "config.json").read_text()))
    ref = reference_for_bundle(a.models, parse_sources(a.train_splits), cfg)
    print(json.dumps({k: v for k, v in ref.items() if k != "psi"}))
    return 0


def _cmd_retune(a) -> int:
    from .paths import PKG
    from .pipeline import RETUNE_SECTIONS, retune
    from .sources import parse_sources

    over = json.loads(Path(a.config).read_text()) if Path(a.config).exists() else json.loads(a.config)
    over = {k: v for k, v in over.items() if k in RETUNE_SECTIONS}
    srcs = parse_sources(a.train_splits)
    ev = pd.read_csv(a.events or PKG / "streams_events.csv", dtype=str)
    t0 = time.time()
    retune(a.models, srcs, ev, over, a.out)
    print(json.dumps({"model_dir": str(Path(a.out).resolve()), "overrides": over, "wall_s": round(time.time() - t0, 1)}))
    return 0


def _cmd_hitattach(a) -> int:
    import pickle

    from .pipeline import attach_hit

    with open(a.hit, "rb") as f:
        state = pickle.load(f)
    if a.tau is not None:
        state["tau"] = None if a.tau.lower() == "none" else float(a.tau)
    if a.nms_order is not None:
        state["nms_order"] = a.nms_order
    out = attach_hit(a.models, state, a.out)
    print(json.dumps({"model_dir": str(Path(out).resolve()), "tau": state["tau"]}))
    return 0


def _cmd_score(a) -> int:
    from .scoring import official_cli, summary

    rep = official_cli(a.pred, a.split, a.out, a.gt)
    print(rep["_cli_stdout"])
    print(json.dumps(summary(rep)))
    return 0


def _cmd_extract(a) -> int:
    from .config import load_config
    from .extract import get_features
    from .sources import parse_sources

    cfg = load_config(a.config)
    for s in parse_sources(a.sources):
        t0 = time.time()
        get_features(s, cfg)
        s.release()
        print(f"{s.sid}: {time.time() - t0:.1f} s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="uwbfall", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fit the pipeline and save a model bundle")
    t.add_argument("--config", default=str(Path(__file__).parent / "configs" / "reference.json"),
                   help="JSON config (partial, merged over the defaults); default: configs/reference.json (= REFERENCE SYSTEM, = DEFAULT_CONFIG)")
    t.add_argument("--train-splits", default="train", help="source spec, e.g. train or train,val")
    t.add_argument("--events", default=None, help="events CSV (default: package streams_events.csv)")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=_cmd_train)
    p = sub.add_parser("predict", help="predict a submission CSV (validated with the official validator)")
    p.add_argument("--models", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default=None)
    p.add_argument("--sources", default=None, help="source spec instead of --split")
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", default=None, help="optional CSV of detector candidates (before replay)")
    p.add_argument("--boundary", default=None,
                   help="JSON (file or inline) overriding the bundle's boundary section at predict time, e.g. "
                        "'{\"gate\": 0.3}' or configs/reference.json (its 'boundary' section is used)")
    p.add_argument("--nms", choices=["off", "time"], default=None,
                   help="override the bundle's final NMS: 'time' = at most one row within decode.nms_gap_s of a "
                        "higher-scored row (reference since round 2, I1), 'off' = none. Segment NMS and the "
                        "P(fall clip) gate need a bundle trained with the clip stage.")
    p.add_argument("--nms-gap", type=float, default=None, help="gap (s) for --nms time (default: config, 4.0)")
    p.set_defaults(fn=_cmd_predict)
    r = sub.add_parser("retune", help="re-run the post-model stages (clip stage, operating point, NMS) of a bundle "
                                      "with a new config, reusing its HGB / TCN models and OOF probabilities")
    r.add_argument("--models", required=True, help="existing bundle (trained on --train-splits)")
    r.add_argument("--config", default=str(Path(__file__).parent / "configs" / "reference.json"),
                   help="JSON (file or inline); only its clip / decode / boundary / report sections are used")
    r.add_argument("--train-splits", default="train")
    r.add_argument("--events", default=None)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=_cmd_retune)
    h = sub.add_parser("hitattach", help="W5 (I3, not promoted): copy a bundle and attach an event hit model "
                                         "(score column) + threshold, from work/I3_score/fit_hit.py")
    h.add_argument("--models", required=True)
    h.add_argument("--hit", required=True, help="pickle {model: hitmodel.HitModel, tau, meta}")
    h.add_argument("--tau", default=None, help="override the stored threshold (float or 'none')")
    h.add_argument("--nms-order", default=None, choices=["score", "detector"],
                   help="final NMS ranks by the hit score (default) or by the detector score (I3 ...N variants)")
    h.add_argument("--out", required=True)
    h.set_defaults(fn=_cmd_hitattach)
    d = sub.add_parser("driftref", help="(re)compute <bundle>/drift_ref.json (label-free drift-monitor reference) "
                                        "from the bundle's training streams; fit does this automatically")
    d.add_argument("--models", required=True)
    d.add_argument("--train-splits", default="train", help="the bundle's training source spec")
    d.set_defaults(fn=_cmd_driftref)
    s = sub.add_parser("score", help="official evaluator CLI (--ignore-flagged)")
    s.add_argument("--pred", required=True)
    s.add_argument("--split", default="val")
    s.add_argument("--gt", default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=_cmd_score)
    x = sub.add_parser("extract", help="fill the feature cache")
    x.add_argument("--sources", required=True)
    x.add_argument("--config", default=None)
    x.set_defaults(fn=_cmd_extract)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

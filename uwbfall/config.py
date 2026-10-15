"""JSON configuration with feature flags.

Every behaviour of the pipeline is driven by one nested dict. ``DEFAULT_CONFIG`` is the current REFERENCE
SYSTEM (= configs/reference.json; round 2: C3G3 + I1 clip gate + time NMS). configs/C3G3.json pins the previous
reference, configs/integration_repro.json the INTEGRATION result (val 0.9646). A user config is a (partial) JSON object deep-merged over the
defaults; unknown keys are rejected so that typos cannot silently change nothing.

Flags reserved for later work packages exist with their "off" value. Turning one on before it is implemented
raises NotImplementedError at fit/predict time. score.hit_model (W5, I3) is implemented but off (not promoted).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "runtime": {
        "threads": 2,               # torch / OpenMP threads (E4 used 2; TCN bits depend on it)
        "deterministic": True,      # torch.use_deterministic_algorithms
        "cache_dir": None,          # None -> $UWBFALL_CACHE or /home/user/data_raw/uwbfall_cache
        "use_cache": True,
    },
    "features": {
        "proto_set": "full",        # physics | structure | full (prototype feature sets)
        "pca_components": 16,       # train-only static-amplitude fingerprint
        "pca_stride": 5,            # PCA fitted on every 5th train block
        "bands": [[0, 5], [5, 13], [13, 22], [22, 40], [40, 53], [53, 69], [69, 100], [100, 184]],
        "use_bands": True,          # E4 "all" = 64 prototype + 48 band channels = 112
        "zones": None,              # W4 D3 hedge (reports/round2/I2.md): list of [radar, bin_a, bin_b) zones, e.g.
                                    # [[0, 40, 69], [1, 5, 22], [2, 5, 22]] -> 16 per-zone static/motion features
        "zones_for": ["hgb"],       # members that get the zone features (TCN input stays 112 channels)
    },
    "labels": {
        "flagged": "positive",      # flagged falls are positives (prototype / E4 behaviour)
        "ignore_clips": "off",      # W4 (reports/round2/I2.md): "zero_weight" = whole fall clips that are generator
                                    # ignore zones (flagged junction-start fall or hard cut) get zero loss weight
        "ignore_members": ["hgb", "tcn"],  # members the zero weights apply to
    },
    "cv": {
        "n_folds": 4,               # GroupKFold by stream on the training sources (operating-point OOF)
    },
    "hgb": {
        "enabled": True,
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": 0,
        "postfall_class": False,    # X3 P0-4: 3-class HGB (other / fall / post-fall-in-clip); P(fall) feeds the blend
        "early_stopping": False,    # PLAN v2 §2.3 (all 300 iterations). "auto" = prototype/E4/INTEGRATION
                                    # (internal 10 % validation split, random_state=0): configs/integration_repro.json
    },
    "tcn": {
        "enabled": True,
        "hidden": 64,
        "stages": 1,
        "dilations": [1, 2, 4, 8, 16, 32, 64],
        "drop": 0.2,
        "in_drop": 0.1,
        "noise": 0.05,
        "lr": 0.001,
        "weight_decay": 0.01,
        "win": 1000,
        "batch": 16,
        "steps": 20,
        "epochs": 8,                # E* chosen by E4 (3-seed CV, min held-out BCE)
        "seeds": 3,
        "cv_seed_base": 100,        # CV fold k, bag member s -> seed cv_seed_base*s + k (E4)
        "final_seed_base": 1000,    # final model member s -> seed final_seed_base + s (E4)
        "seed_offset": 0,           # added to every CV and final TCN seed (0 = E4 seeds). Re-seeds a whole bag;
                                    # cv_seed_base alone cannot (member 0 of CV fold k always gets seed k).
        "clip_z": 6.0,
    },
    "blend": {
        "w_tcn": 0.75,              # p = w*TCN + (1-w)*HGB (chosen by E4 on train OOF)
    },
    "decode": {
        # prototype post-processing grid; the best point is chosen on training OOF (official evaluator code)
        "grid": {"thr": [0.3, 0.4, 0.5, 0.6, 0.7], "smooth": [1, 3, 5], "min_dur": [0.4, 0.8], "gap": [0.2, 0.6]},
        "fixed": None,              # [thr, smooth, min_dur, gap] to skip the OOF search
        "one_event_per_clip": "time",  # W3 hard 1-D NMS on the final (replayed) rows: False | "time" | "segment".
                                    # "time" promoted in round 2 (reports/round2/I1.md: REF->N4 P=1.000, 4/4 folds)
        "nms_gap_s": 4.0,           # "time": drop a row within this gap (s) of a higher-scored row of the stream
                                    # (4 = smallest gap reaching the best pooled train-OOF rank F1, nms_oof.py)
        "nms_max_span_s": 18.7,     # "segment": rows share a segment only if their span <= this (longest fall clip)
    },
    "clip": {
        # W3 / X2 P0-2: variance-normalised junction detector; threshold chosen on the training streams for
        # recall of fall-clip junctions
        "junction_W": 5,
        "junction_recall": 0.95,
        "junction_min_sep_s": 1.0,
        # W3 / X2 P1-1, X3 P0-3: LightGBM segment classifier (adl / fall clip / autre-empty) and gating of the
        # frame posteriors by P(fall clip) of the block's junction-to-junction segment
        "segclf": {"enabled": True, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15,
                   "min_child_samples": 10, "colsample_bytree": 0.8,
                   "features": "nofp",   # all | nofp | norep | phys | phys+sp3 (chosen on subject-grouped train OOF)
                   "oof_groups": "subject"},  # OOF of P(fall clip) on training streams for the decode tuning
        "gate": "mult",             # none | mult | sqrt   (p * g, p * sqrt(g), g = floor + (1-floor) P(fall clip))
        "gate_floor": 0.7,          # chosen on train OOF among {0, .3, .5, .7}; gate promoted in round 2
                                    # (reports/round2/I1.md: N4->G7N4 P=0.987, 3/4 folds, 3 TCN bags)
    },
    "boundary": {
        "method": "replay",         # replay | none
        "crops": ["loc2_3d", "loc2_5d", "det2_3d", "det2_5d", "detq2_3d", "detq2_5d"],
        "gate": 0.3,                # agreement gate tIoU(crop replay, detector); 0.1 = E2 config B. 0.3 promoted
                                    # on SR-CV (reports/round2/EVAL.md: P=1.000, 4/4 folds, 3 TCN bags)
        "min_ok": 1,
        "agg": "median",            # median of agreeing crops (E2) | cluster: median of the largest agreeing cluster (I2)
        "cluster_iou": 0.8,         # "cluster": crops within this tIoU of the medoid crop form its cluster
        "fallback": "detector",     # detector | contain: if no crop passes the gate, median of the crops lying mostly
        "contain_frac": 0.8,        #   inside the detector interval (intersection >= frac x crop length) (I2)
        "selector": None,           # None | "lgbm": W4 boundary-hypothesis selector (I2; boundary.select)
        "dmin": 1.0,                # duration floor, extends the end only
        "ds": 0.0, "de": 0.0, "pds": 0.0, "pde": 0.0,
        "dedup": True,              # False: replay on all frames for every crop (crops ending in "d" use distinct frames)
        "junction_q": 97,
        "junction_q_recall": 95,
    },
    "score": {
        "hit_model": None,          # W5 (I3): None = peak smoothed blend probability | "lgbm": event hit model
                                    # (uwbfall.hitmodel; needs boundary.method=replay and a bundle with hit.pkl,
                                    # attached by `python -m uwbfall hitattach`); the final NMS then keeps the
                                    # higher hit score. Not promoted (reports/round2/I3.md).
        "threshold": None,          # None | float tau: drop final rows with hit score < tau (after NMS). Stored by
                                    # hitattach (nested choice on SR-CV OOF, or the fixed F1*/2 rule)
        "nms_order": "score",       # "score": the final NMS keeps the higher hit score | "detector": it keeps the
                                    # higher detector score (kept rows = the reference's; the hit score then acts
                                    # only through the threshold cut; I3 variant ...N). Only used with hit_model.
        "seeds": 3,                 # LightGBM seeds averaged per theta
        "inner_folds": 5,
        "tau_grid": [0.0, 0.6, 0.02],
        "lgbm": {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 20},
    },
    "transductive": {
        "adl_duplicate_mask": False,   # cross-stream ADL duplicate mask: must stay OFF (rules)
    },
    "report": {
        "oof_replay": True,         # also replay boundaries on the training OOF candidates (needs raw crops,
                                    # ~1-2 min); stores oof_final.csv in the bundle for SR-CV pooling
    },
}

NOT_IMPLEMENTED = {
    ("transductive", "adl_duplicate_mask"): False,
}


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise KeyError(f"unknown config key {path + k!r}")
        if isinstance(base[k], dict) and isinstance(v, dict) and k != "grid":
            out[k] = _merge(base[k], v, path + k + ".")
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path_or_dict: str | Path | dict | None = None) -> dict:
    if path_or_dict is None:
        over = {}
    elif isinstance(path_or_dict, dict):
        over = path_or_dict
    else:
        over = json.loads(Path(path_or_dict).read_text())
    return _merge(DEFAULT_CONFIG, over)


def check_implemented(cfg: dict) -> None:
    for (sec, key), off in NOT_IMPLEMENTED.items():
        if cfg[sec][key] != off:
            raise NotImplementedError(f"config {sec}.{key}={cfg[sec][key]!r} is reserved but not implemented yet")
    if cfg["boundary"].get("selector") is not None:
        raise NotImplementedError("boundary.selector is experiment-only (work/I2_boundaries/sel.py); it failed the "
                                  "promotion rule and is not wired into fit/predict")
    if cfg["score"]["hit_model"] not in (None, "lgbm"):
        raise ValueError("score.hit_model must be null or 'lgbm'")
    if cfg["score"]["hit_model"] is not None and cfg["boundary"]["method"] != "replay":
        raise ValueError("score.hit_model needs boundary.method='replay' (replay features)")
    if cfg["score"]["threshold"] is not None and (cfg["score"]["hit_model"] is None
                                                  or not 0 <= float(cfg["score"]["threshold"]) <= 1):
        raise ValueError("score.threshold needs score.hit_model and must lie in [0, 1]")
    if cfg["score"]["nms_order"] not in ("score", "detector"):
        raise ValueError("score.nms_order must be 'score' or 'detector'")
    if cfg["boundary"]["method"] not in ("replay", "none"):
        raise ValueError("boundary.method must be 'replay' or 'none'")
    if cfg["decode"]["one_event_per_clip"] not in (False, "time", "segment"):
        raise ValueError("decode.one_event_per_clip must be false, 'time' or 'segment'")
    if cfg["clip"]["gate"] not in ("none", "mult", "sqrt"):
        raise ValueError("clip.gate must be none | mult | sqrt")
    if cfg["clip"]["gate"] != "none" and not cfg["clip"]["segclf"]["enabled"]:
        raise ValueError("clip.gate needs clip.segclf.enabled")
    if cfg["labels"]["ignore_clips"] not in ("off", "zero_weight"):
        raise ValueError("labels.ignore_clips must be 'off' or 'zero_weight'")
    if not (cfg["hgb"]["enabled"] or cfg["tcn"]["enabled"]):
        raise ValueError("at least one of hgb / tcn must be enabled")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

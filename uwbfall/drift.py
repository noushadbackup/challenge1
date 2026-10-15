"""Label-free drift monitors (PLAN v2 §3 / §6: physics-only switch triggered by drift monitors).

At predict time, for the predicted streams, compare with what the bundle saw on its training streams:

  * predicted events per hour            (reference: the bundle's own OOF final predictions, oof_final.csv)
  * detected junctions per hour          (variance-normalised junction statistic, same threshold as training)
  * share of predictions near a junction (prediction start within `NEAR_S` s after a detected junction)
  * repeated-frame share                 (sample-and-hold repeats; generator/writer artefact, 23-27 % in the package)
  * per-radar feature PSI vs training    (block-level log motion energy, log frame-difference energy, repeated-frame
                                          fraction; 10 training-quantile bins) + the per-radar median raw-energy level
                                          (scale / writer change; raw-energy *shape* is room-dependent, see FEATS)

Everything is computed within the predicted streams (no labels, no cross-stream content matching).
The reference is stored in ``<bundle>/drift_ref.json`` (written by ``fit`` and by ``python -m uwbfall driftref``).

Alert rules (heuristic, documented in SUBMISSION.md; a value outside the band prints ALERT):
  events/h        outside [0.5, 1.6] x reference OOF rate
  junctions/h     outside [0.6, 1.6] x reference
  near-junction   below reference - 0.20
  repeat share    outside [0.5, 1.5] x reference
  PSI             pooled PSI > 0.25 on any (radar, feature)  (0.10-0.25 = WATCH)
  raw level       per-radar median log10 raw energy outside the training per-stream range +- 1 decade
Two or more ALERT lines -> the CLI recommends the physics-only fallback (configs/fallback_physics.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import clip as CL
from .paths import STEP_S

NEAR_S = 5.0            # train13 OOF predictions: 88 % start within 5 s after a detected junction, random times 33 %
                        # (fall onsets sit 1-4 s into their clip); a drop means detections decoupled from clip structure
FEATS = ("log_emot", "log_ediff", "rep_frac")   # PSI features. Raw (static) energy is NOT in the PSI: its shape is
# room / clutter dependent (val vs train13: PSI 0.29-1.16 even after removing the stream median, and train streams
# among themselves reach 0.56), so it would alarm on every new subject. Its level is checked instead (LEVEL_TOL).
LEVEL_TOL = 1.0         # decades: per-radar median raw-energy level outside the training per-stream range +- 1
REF_FILE = "drift_ref.json"
PSI_WATCH, PSI_ALERT = 0.10, 0.25
BANDS = {"events_per_h": (0.5, 1.6), "junctions_per_h": (0.6, 1.6), "repeat_share": (0.5, 1.5)}
NEAR_DROP = 0.20


def block_feats(F: dict) -> dict[str, np.ndarray]:
    """(3, n_blocks) arrays per monitored feature, 0.2 s blocks (10 frames)."""
    T = F["e_raw"].shape[1]
    n = T // 10

    def bm(x):
        return x[:, : n * 10].astype(np.float64).reshape(3, n, 10).mean(2)

    rep = np.zeros_like(F["e_diff"], dtype=np.float64)
    rep[:, 1:] = F["e_diff"][:, 1:] == 0
    le = np.log10(bm(F["e_raw"]) + 1e-12)
    return {"log_eraw_level": np.median(le, 1),
            "log_emot": np.log10(bm(F["e_mot"]) + 1e-12),
            "log_ediff": np.log10(bm(F["e_diff"]) + 1e-12), "rep_frac": bm(rep)}


def junction_times(F: dict, thr: float, W: int = 5, min_sep_s: float = 1.0) -> np.ndarray:
    return CL.junction_peaks(CL.junction_stat(F["prof_02"], W), thr, min_sep_s) * STEP_S


def near_share(pred: pd.DataFrame, peaks: dict) -> tuple[int, int]:
    """(# predictions whose start lies within NEAR_S s after a detected junction, # predictions)."""
    near = tot = 0
    for sid, g in pred.groupby("stream_id"):
        pk = np.sort(np.asarray(peaks.get(sid, []), float))
        for t in g.t_start.values.astype(float):
            tot += 1
            i = np.searchsorted(pk, t, side="right") - 1
            near += bool(i >= 0 and t - pk[i] <= NEAR_S)
    return near, tot


def stream_stats(F: dict, thr: float) -> dict:
    """Per-stream monitor inputs: duration, junction times, repeated-frame share, block features (subsampled)."""
    bf = block_feats(F)
    T = F["e_raw"].shape[1]
    return {"hours": T / 50.0 / 3600.0, "peaks": junction_times(F, thr),
            "repeat_share": float(CL.repeat_flags(F["e_diff"]).mean()), "bf": bf}


def _hist(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1).astype(float)
    return h / max(h.sum(), 1.0)


def psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-4) -> float:
    p, q = np.maximum(p, eps), np.maximum(q, eps)
    return float(np.sum((q - p) * np.log(q / p)))


def _edges(x: np.ndarray, feat: str) -> np.ndarray:
    if feat == "rep_frac":                       # discrete 0, .1, ..., 1
        return np.arange(0.05, 1.0, 0.1)
    return np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9)))


def build_reference(stats: dict[str, dict], oof_final: pd.DataFrame | None, thr: float, thr_source: str) -> dict:
    """stats: {sid: stream_stats(...)} of the training streams."""
    hours = sum(s["hours"] for s in stats.values())
    ref = {"junction_thr": thr, "junction_thr_source": thr_source, "n_streams": len(stats), "hours": hours,
           "junctions_per_h": sum(len(s["peaks"]) for s in stats.values()) / hours,
           "repeat_share": float(np.mean([s["repeat_share"] for s in stats.values()])),
           "repeat_share_range": [float(min(s["repeat_share"] for s in stats.values())),
                                  float(max(s["repeat_share"] for s in stats.values()))],
           "psi": {}}
    lv = np.array([s["bf"]["log_eraw_level"] for s in stats.values()])  # (streams, 3)
    ref["eraw_level_range"] = [[float(lv[:, r].min()), float(lv[:, r].max())] for r in range(3)]
    per_stream_j = [len(s["peaks"]) / s["hours"] for s in stats.values()]
    ref["junctions_per_h_range"] = [float(min(per_stream_j)), float(max(per_stream_j))]
    if oof_final is not None and len(oof_final):
        ref["events_per_h"] = len(oof_final) / hours
        nr, nt = near_share(oof_final, {k: s["peaks"] for k, s in stats.items()})
        ref["near_junction_share"] = nr / max(nt, 1)
        per = oof_final.groupby("stream_id").size()
        ref["events_per_h_range"] = [float(min(per.get(k, 0) / s["hours"] for k, s in stats.items())),
                                     float(max(per.get(k, 0) / s["hours"] for k, s in stats.items()))]
    for f in FEATS:
        for r in range(3):
            pooled = np.concatenate([s["bf"][f][r] for s in stats.values()])
            e = _edges(pooled, f)
            p = _hist(pooled, e)
            # typical per-stream PSI of a training stream against the pooled training histogram (content variation)
            ps = [psi(p, _hist(s["bf"][f][r], e)) for s in stats.values()]
            ref["psi"][f"{f}/r{r}"] = {"edges": e.tolist(), "p": p.tolist(),
                                       "train_stream_psi_median": float(np.median(ps)),
                                       "train_stream_psi_max": float(np.max(ps))}
    return ref


def reference_for_bundle(bundle_dir, sources, cfg: dict, log=print) -> dict:
    """Compute and write <bundle>/drift_ref.json from the bundle's training sources (features from the cache)."""
    import pickle

    from .clip import composition_of
    from .extract import get_features
    b = Path(bundle_dir)
    thr, src = None, ""
    if (b / "clip.pkl").exists():
        with open(b / "clip.pkl", "rb") as f:
            thr, src = float(pickle.load(f)["thr"]), "bundle clip stage"
    stats, jst = {}, {}
    for s in sorted(sources, key=lambda s: s.sid):
        F = get_features(s, cfg)
        if thr is None:
            jst[s.sid] = (F, CL.junction_stat(F["prof_02"], cfg["clip"]["junction_W"]))
        else:
            stats[s.sid] = stream_stats(F, thr)
        s.release()
    if thr is None:   # bundle without a clip stage: same threshold rule as the clip stage (0.95 fall-junction recall)
        thr = CL.junction_threshold({k: v[1] for k, v in jst.items()},
                                    {k: composition_of(s) for s in sources for k in [s.sid] if k in jst},
                                    cfg["clip"]["junction_recall"])
        src = "computed (clip-stage rule) on the training streams"
        stats = {k: stream_stats(v[0], thr) for k, v in jst.items()}
    oof = pd.read_csv(b / "oof_final.csv") if (b / "oof_final.csv").exists() else None
    ref = build_reference(stats, oof, thr, src)
    (b / REF_FILE).write_text(json.dumps(ref, indent=1))
    log(f"drift reference written: {b / REF_FILE} (junctions/h {ref['junctions_per_h']:.1f}, "
        f"events/h {ref.get('events_per_h', float('nan')):.1f}, repeat share {ref['repeat_share']:.3f})")
    return ref


def load_reference(bundle_dir) -> dict | None:
    p = Path(bundle_dir) / REF_FILE
    return json.loads(p.read_text()) if p.exists() else None


def report(ref: dict | None, stats: dict[str, dict], pred: pd.DataFrame) -> dict:
    """Monitor values for the predicted streams vs the reference, with OK / WATCH / ALERT flags."""
    hours = sum(s["hours"] for s in stats.values())
    peaks = {k: s["peaks"] for k, s in stats.items()}
    nr, nt = near_share(pred, peaks)
    cur = {"n_streams": len(stats), "hours": hours, "events_per_h": len(pred) / max(hours, 1e-9),
           "junctions_per_h": sum(len(p) for p in peaks.values()) / max(hours, 1e-9),
           "near_junction_share": nr / max(nt, 1) if nt else float("nan"),
           "repeat_share": float(np.mean([s["repeat_share"] for s in stats.values()])) if stats else float("nan"),
           "per_stream": {k: {"events_per_h": float((pred.stream_id == k).sum() / s["hours"]),
                              "junctions_per_h": len(s["peaks"]) / s["hours"],
                              "repeat_share": s["repeat_share"]} for k, s in stats.items()}}
    checks = []
    if ref is not None:
        for key, (lo, hi) in BANDS.items():
            if key in ref and key in cur:
                r = cur[key] / max(ref[key], 1e-9)
                checks.append({"monitor": key, "value": cur[key], "ref": ref[key], "ratio": r,
                               "flag": "OK" if lo <= r <= hi else "ALERT", "rule": f"ratio in [{lo}, {hi}]"})
        if "near_junction_share" in ref and nt:
            v = cur["near_junction_share"]
            checks.append({"monitor": "near_junction_share", "value": v, "ref": ref["near_junction_share"],
                           "flag": "OK" if v >= ref["near_junction_share"] - NEAR_DROP else "ALERT",
                           "rule": f">= ref - {NEAR_DROP}"})
        for r, (lo, hi) in enumerate(ref.get("eraw_level_range", [])):
            v = float(np.median([s["bf"]["log_eraw_level"][r] for s in stats.values()]))
            checks.append({"monitor": f"raw energy level r{r}", "value": v, "ref": (lo + hi) / 2,
                           "flag": "OK" if lo - LEVEL_TOL <= v <= hi + LEVEL_TOL else "ALERT",
                           "rule": f"log10 in train range [{lo:.2f}, {hi:.2f}] +- {LEVEL_TOL}"})
        cur["psi"] = {}
        for key, R in ref["psi"].items():
            f, r = key.split("/r")
            x = np.concatenate([s["bf"][f][int(r)] for s in stats.values()])
            v = psi(np.asarray(R["p"]), _hist(x, np.asarray(R["edges"])))
            cur["psi"][key] = v
            checks.append({"monitor": f"psi {key}", "value": v, "ref": R["train_stream_psi_median"],
                           "flag": "OK" if v <= PSI_WATCH else ("WATCH" if v <= PSI_ALERT else "ALERT"),
                           "rule": f"pooled PSI <= {PSI_WATCH} OK, <= {PSI_ALERT} WATCH"})
    n_alert = sum(c["flag"] == "ALERT" for c in checks)
    return {"current": cur, "checks": checks, "n_alert": n_alert, "has_reference": ref is not None,
            "recommend_fallback": n_alert >= 2}


def format_report(rep: dict) -> str:
    c = rep["current"]
    lines = [f"DRIFT MONITORS ({c['n_streams']} streams, {c['hours']:.2f} h)"
             + ("" if rep["has_reference"] else "  [no drift_ref.json in bundle: values only]")]
    if not rep["checks"]:
        lines.append(f"  events/h {c['events_per_h']:.1f}  junctions/h {c['junctions_per_h']:.1f}  "
                     f"near-junction share {c['near_junction_share']:.3f}  repeat share {c['repeat_share']:.3f}")
    for k in rep["checks"]:
        lines.append(f"  {k['flag']:5s} {k['monitor']:<24s} {k['value']:9.3f}  (train ref {k['ref']:.3f}; {k['rule']})")
    for sid, s in c["per_stream"].items():
        lines.append(f"  stream {sid}: events/h {s['events_per_h']:.1f}, junctions/h {s['junctions_per_h']:.1f}, "
                     f"repeat share {s['repeat_share']:.3f}")
    lines.append(f"  -> {rep['n_alert']} ALERT(s); "
                 + ("RECOMMEND physics-only fallback (see SUBMISSION.md)" if rep["recommend_fallback"]
                    else "no fallback recommended"))
    return "\n".join(lines)

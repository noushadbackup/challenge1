"""TAL member: alone and fused with the base system (final_norep). Every choice is made on the bundle's
TRAINING-stream OOF (inner folds); test streams / val are only predicted.

    python3 work/R3_tal/fuse.py alone  <tag> [bundles]      TAL alone: (tau, nms gap) on OOF -> results/<tag>_alone
    python3 work/R3_tal/fuse.py screen <tag> [bundles]      blend screen on OOF (detector level) -> json
    python3 work/R3_tal/fuse.py blend  <tag> <variant> <member> <lam|auto> [bundles]
                                                            fake bundle + retune + predict -> results/<variant>
    python3 work/R3_tal/fuse.py post   <tag> [bundles]      boundary-averaging / TAL-support gate on base rows
    python3 work/R3_tal/fuse.py score  <variant> [base]     pooled SR-CV + promote vs base (norep)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
import srcv  # noqa: E402
from srcv import core as C  # noqa: E402
import uwbfall  # noqa: E402
from uwbfall import pipeline as P  # noqa: E402
from uwbfall import clip as CL  # noqa: E402
from uwbfall.config import load_config  # noqa: E402
from uwbfall.decode import tune, decode_streams  # noqa: E402
from uwbfall.scoring import gt_records, score_inproc, summary  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import tal  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
BASE_RUNS = Path("/home/user/data_raw/srcv/runs/strict/norep")
BASE_RES = HERE.parents[1] / "work" / "STRICT" / "results" / "norep"
OUT = Path("/home/user/data_raw/r3/tal")
BUNDLES = ["fold_0", "fold_1", "fold_2", "fold_3", "official_train"]
COLS = ["stream_id", "t_start", "t_end", "score", "t_alarm"]


def train_events(b):
    return C.real_events("train") if b == "official_train" else C.events(int(b[-1]), "train")


def train_sources(b):
    return uwbfall.split_sources("train") if b == "official_train" else srcv.pipeline_sources(int(b[-1]), "train")


def test_sources(b):
    return uwbfall.split_sources("val") if b == "official_train" else srcv.pipeline_sources(int(b[-1]), "test")


def out_name(b):
    return "val.csv" if b == "official_train" else f"{b}.csv"


def gt_dur(b, tr):
    return gt_records(train_events(b), ignore_flagged=True), {s: v["dur"] for s, v in tr.items()}


def rows_df(segs: pd.DataFrame, dur: dict) -> pd.DataFrame:
    d = segs.copy()
    d["t_start"] = d.t_start.clip(lower=0).round(3)
    d["t_end"] = np.minimum(d.t_end.values, d.stream_id.map(dur).values).round(3)
    d = d[d.t_end > d.t_start]
    d["t_alarm"] = d.t_start
    return d[COLS].reset_index(drop=True)


# ------------------------------------------------------------------------------------------- TAL alone
TAUS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
GAPS = [-1, 0, 2, 4]


def alone_rows(segs, dur, tau, gap):
    d = rows_df(segs[segs.score >= tau], dur)
    return CL.nms_time(d, gap).reset_index(drop=True)[COLS] if gap >= 0 else d


def alone(tag, bundles):
    res = RES / f"{tag}_alone"
    res.mkdir(parents=True, exist_ok=True)
    info = {}
    for b in bundles:
        tr, te = tal.load(b)
        gt, dur = gt_dur(b, tr)
        o = OUT / "out" / tag / b
        oof = pd.read_csv(o / "oof_segs.csv")
        best = None
        for tau in TAUS:
            for gap in GAPS:
                sc = score_inproc(alone_rows(oof, dur, tau, gap), gt, dur)["ranking_score"]["f1_mean"]
                if best is None or sc > best[0]:
                    best = (sc, tau, gap)
        tst = pd.read_csv(o / "test_segs.csv")
        alone_rows(tst, {s: v["dur"] for s, v in te.items()}, best[1], best[2]).to_csv(res / out_name(b), index=False)
        info[b] = {"oof_rank_f1": best[0], "tau": best[1], "gap": best[2]}
        print(b, info[b], flush=True)
    (res / "choice.json").write_text(json.dumps(info, indent=1))


# ------------------------------------------------------------------------------------------- blend
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MEMBERS = ["aux", "pmax", "cov"]


def oof_parts(tag, b):
    tr, _ = tal.load(b)
    z = np.load(OUT / "out" / tag / b / "oof.npz")
    sids = list(tr)   # bundle order = sorted sids, same as oof.npz of the base bundle
    base = {s: 0.75 * tr[s]["oof_tcn"].astype(np.float64) + 0.25 * tr[s]["oof_hgb"].astype(np.float64) for s in sids}
    fac = {s: tr[s]["oof_blend"] / np.maximum(base[s], 1e-12) for s in sids}
    return tr, z, sids, base, fac


def screen(tag, bundles):
    out = {}
    for b in bundles:
        tr, z, sids, base, fac = oof_parts(tag, b)
        gt, dur = gt_dur(b, tr)
        cfg = load_config(json.loads((BASE_RUNS / b / "config.json").read_text()))
        out[b] = {}
        for mem in MEMBERS:
            for lam in LAMS:
                if lam == 0 and mem != MEMBERS[0]:
                    continue
                probs = {s: fac[s] * ((1 - lam) * base[s] + lam * z[f"{mem}__{s}"]) for s in sids}
                sc, g, _ = tune(probs, gt, dur, cfg)
                key = "base" if lam == 0 else f"{mem}_{lam}"
                out[b][key] = {"rank": sc, "op": list(g)}
                print(b, key, round(sc, 4), g, flush=True)
    p = HERE / f"screen_{tag}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1))


def make_fake(tag, b, member, lam, dst: Path):
    """Copy of the base bundle whose oof.npz carries the TAL-blended OOF (as oof_tcn' with 0.75 t' + 0.25 h =
    (1 - lam) (0.75 t + 0.25 h) + lam * tal), so that retune() re-fits clip gate / operating point / replay / NMS."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE_RUNS / b, dst)
    tr, zt, sids, base, _fac = oof_parts(tag, b)
    zb = dict(np.load(BASE_RUNS / b / "oof.npz"))
    tal_oof = np.zeros(len(zb["sid"]))
    for s in sids:
        tal_oof[zb["sid"] == s] = zt[f"{member}__{s}"]
    h = zb["oof_hgb"]
    tgt = (1 - lam) * (0.75 * zb["oof_tcn"].astype(np.float64) + 0.25 * h) + lam * tal_oof
    zb["oof_tcn"] = (tgt - 0.25 * h) / 0.75
    np.savez_compressed(dst / "oof.npz", **zb)


def patched_stream_probs(B, F):
    """uwbfall.pipeline.stream_probs + the TAL member blended in before the clip gate (single stream)."""
    from uwbfall.features import stream_blocks, design
    from uwbfall.models import standardise, predict_seq
    t = B.get("tal")
    if t is None:
        return ORIG_STREAM_PROBS(B, F)
    cfg = B["cfg"]
    blk = stream_blocks(F, cfg)
    xp, xb, cols = design(blk, B["pca"], cfg)
    p_hgb = B["hgb"].predict_proba(xp)[:, 1]
    X = standardise(np.hstack([xp.astype(np.float32), xb]), B["mu"], B["sd"], cfg["tcn"]["clip_z"])
    p_tcn = np.mean([predict_seq(m, X) for m in B["tcn"]], 0)
    p = P._blend(cfg, p_tcn, p_hgb)
    outs = [tal.stream_outputs(m, X, blk["n"] * tal.STEP) for m in t["models"]]
    aux, pmax = np.mean([o[1] for o in outs], 0), np.mean([o[2] for o in outs], 0)
    segs = outs[0][0] if len(outs) == 1 else tal.soft_nms(np.vstack([o[0] for o in outs]))
    mem = {"aux": aux, "pmax": pmax, "cov": tal.coverage(segs, blk["n"])}[t["member"]]
    p = (1 - t["lam"]) * p + t["lam"] * mem
    parts = {"hgb": p_hgb, "tcn": p_tcn, "peaks": None, "tal": mem}
    if B.get("clip") is not None:
        fpc = [i for i, c in enumerate(cols) if c.startswith("fp") and not c.startswith("fpseg")]
        sg, tab, pk_t, _ = CL.stream_segments(F, blk["base"], xp[:, fpc], B["clip"]["thr"], cfg)
        parts["peaks"] = pk_t
        if B["clip"]["segclf"] is not None and cfg["clip"]["gate"] != "none":
            pf = B["clip"]["segclf"].predict_proba(tab[B["clip"]["features"]])[:, 1]
            parts["gate"] = CL.block_gate(sg, pf, blk["n"])
            p = CL.gate_probs(p, parts["gate"], cfg["clip"]["gate"], cfg["clip"]["gate_floor"])
    return p, parts


ORIG_STREAM_PROBS = P.stream_probs


def blend(tag, variant, member, lam, bundles):
    res = RES / variant
    res.mkdir(parents=True, exist_ok=True)
    scr = json.loads((HERE / f"screen_{tag}.json").read_text()) if lam == "auto" else {}
    if lam == "autoF":   # final-level (replay + NMS) training-OOF rank F1 from screen_final
        scr = {b: {k: {"rank": v["final"]} for k, v in d.items()}
               for b, d in json.loads((HERE / f"screenF_{tag}.json").read_text()).items()}
    P.stream_probs = patched_stream_probs
    info = {}
    for b in bundles:
        if lam in ("auto", "autoF"):   # per-bundle choice on its own training OOF (detector-level rank F1)
            cand = {k: v["rank"] for k, v in scr[b].items() if member == "any" or k == "base" or k.startswith(member)}
            key = max(cand, key=lambda k: (cand[k], k == "base"))
            m_, l_ = ("aux", 0.0) if key == "base" else (key.rsplit("_", 1)[0], float(key.rsplit("_", 1)[1]))
        else:
            m_, l_ = member, float(lam)
        if l_ == 0.0:   # base chosen on training OOF: the base system's own rows
            shutil.copy(BASE_RES / out_name(b), res / out_name(b))
            info[b] = {"member": None, "lam": 0.0, "note": "base chosen on training OOF"}
            print(b, json.dumps(info[b]), flush=True)
            continue
        fake = OUT / "bundles" / variant / f"{b}_src"
        dst = OUT / "bundles" / variant / b
        make_fake(tag, b, m_, l_, fake)
        bdir = P.retune(fake, train_sources(b), train_events(b), {}, dst)
        shutil.rmtree(fake)
        B = P.load_bundle(bdir)
        tr, _ = tal.load(b)
        models = []
        for f in sorted((OUT / "out" / tag / b).glob("tal*.pt")):
            model = tal.TALNet(112, json.loads((OUT / "out" / tag / b / "args.json").read_text())["h"])
            model.load_state_dict(torch.load(f, weights_only=True))
            model.eval()
            models.append(model)
        B["tal"] = {"models": models, "member": m_, "lam": l_}
        uwbfall.predict(B, test_sources(b)).to_csv(res / out_name(b), index=False)
        meta = json.loads((bdir / "meta.json").read_text())
        info[b] = {"member": m_, "lam": l_, "op": meta["postproc"], "oof": meta["oof"]}
        print(b, json.dumps(info[b]), flush=True)
    (res / "choice.json").write_text(json.dumps(info, indent=1))
    P.stream_probs = ORIG_STREAM_PROBS


# ------------------------------------------------------------------------------------------- final-level screen
class ReplayCache:
    """Annotator replay of OOF candidates, memoised per (stream, t_start, t_end): the replay of a candidate only
    depends on its bounds and its own stream, so blends that share candidates share the (slow) replay."""
    def __init__(self, b, cfg):
        self.src = {s.sid: s for s in train_sources(b)}
        self.cfg = cfg
        self.rows = {}

    def final(self, cand: pd.DataFrame, durations) -> pd.DataFrame:
        from uwbfall.extract import get_features
        from uwbfall.boundary import replay_stream, refine, to_preds
        out = []
        for sid, c in cand.groupby("stream_id", sort=True):
            keys = list(zip(c.t_start.values, c.t_end.values))
            new = c[[(sid, k) not in self.rows for k in keys]]
            if len(new):
                s = self.src[sid]
                F = get_features(s, self.cfg)
                D = replay_stream(s, F["prof_02"], new.reset_index(drop=True), self.cfg)
                s.release()
                for i, r in D.iterrows():
                    self.rows[(sid, (r.t_start, r.t_end))] = r
            D = pd.DataFrame([self.rows[(sid, k)] for k in keys]).reset_index(drop=True)
            D["score"] = c.score.values
            a, e, _ = refine(D, self.cfg["boundary"])
            out.append(to_preds(D, a, e, durations))
        f = pd.concat(out, ignore_index=True) if out else cand.iloc[:0]
        return P._nms(f, self.cfg, {})


SC_LAMS = [0.15, 0.3, 0.45]


def screen_final(tag, bundles, members=("pmax", "aux", "cov")):
    """Final-level (after replay + time NMS) training-OOF rank F1 of each (member, lam); operating point re-tuned
    on the detector-level blended OOF exactly as retune() does (the clip gate is the base bundle's OOF gate)."""
    p = HERE / f"screenF_{tag}.json"
    allres = json.loads(p.read_text()) if p.exists() else {}
    for b in bundles:
        tr, z, sids, base, fac = oof_parts(tag, b)
        gt, dur = gt_dur(b, tr)
        cfg = load_config(json.loads((BASE_RUNS / b / "config.json").read_text()))
        rc = ReplayCache(b, cfg)
        res = allres.get(b, {})
        opts = [("base", None, 0.0)] + [(f"{m}_{l}", m, l) for m in members for l in SC_LAMS]
        for key, mem, lam in opts:
            if key in res:
                continue
            probs = {s: fac[s] * ((1 - lam) * base[s] + (lam * z[f"{mem}__{s}"] if mem else 0)) for s in sids}
            sc, g, _ = tune(probs, gt, dur, cfg)
            fin = rc.final(decode_streams(probs, g), dur)
            sm = summary(score_inproc(fin, gt, dur))
            res[key] = {"det": sc, "final": sm["rank_f1"], "op": list(g), "summary": sm}
            print(b, key, round(sc, 4), sm, flush=True)
            allres[b] = res
            p.write_text(json.dumps(allres, indent=1))


# ------------------------------------------------------------------------------------------- post-hoc on rows
def match_tal(rows: pd.DataFrame, segs: pd.DataFrame, min_iou: float):
    """For each row: best-tIoU TAL segment of the same stream (start, end, score, iou) and max TAL score of any
    segment overlapping the row (support)."""
    out = np.full((len(rows), 5), np.nan)
    by = {s: g[["t_start", "t_end", "score"]].values for s, g in segs.groupby("stream_id")}
    for i, r in enumerate(rows.itertuples()):
        S = by.get(r.stream_id)
        if S is None or not len(S):
            out[i, 4] = 0
            continue
        iou = tal.tiou_1n((r.t_start, r.t_end), S)
        j = int(np.argmax(iou))
        ov = iou > 0
        out[i, 4] = S[ov, 2].max() if ov.any() else 0.0
        if iou[j] >= min_iou:
            out[i, :4] = [S[j, 0], S[j, 1], S[j, 2], iou[j]]
    return out


def post_apply(rows, segs, beta, min_iou, tau_sup, dur):
    m = match_tal(rows, segs, min_iou)
    r = rows.copy()
    ok = ~np.isnan(m[:, 0])
    r.loc[ok, "t_start"] = ((1 - beta) * r.t_start.values[ok] + beta * m[ok, 0]).round(3)
    r.loc[ok, "t_end"] = ((1 - beta) * r.t_end.values[ok] + beta * m[ok, 1]).round(3)
    r = r[m[:, 4] >= tau_sup]
    r["t_start"] = r.t_start.clip(lower=0)
    r["t_end"] = np.minimum(r.t_end.values, r.stream_id.map(dur).values)
    r = r[r.t_end > r.t_start]
    r["t_alarm"] = r.t_start
    return r[COLS].reset_index(drop=True)


def post(tag, bundles):
    """Two post-hoc fusions on the base final rows, tuned on the base bundle's OOF final rows (oof_final.csv)
    against the TAL inner-fold OOF segments: (a) boundary averaging (beta, min tIoU); (b) support gate tau_sup."""
    grids = {"bavg": [(be, mi, 0.0) for be in (0.0, 0.1, 0.2, 0.3, 0.5) for mi in (0.3, 0.5, 0.7)],
             "tgate": [(0.0, 1.0, ts) for ts in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15)]}
    info = {}
    for name, grid in grids.items():
        res = RES / f"{tag}_{name}"
        res.mkdir(parents=True, exist_ok=True)
        info[name] = {}
        for b in bundles:
            tr, te = tal.load(b)
            gt, dur = gt_dur(b, tr)
            base_oof = pd.read_csv(BASE_RUNS / b / "oof_final.csv")
            oof = pd.read_csv(OUT / "out" / tag / b / "oof_segs.csv")
            best = None
            for g in grid:
                sc = score_inproc(post_apply(base_oof, oof, *g, dur), gt, dur)["ranking_score"]["f1_mean"]
                if best is None or sc > best[0] + 1e-9:
                    best = (sc, g)
            base_sc = score_inproc(base_oof, gt, dur)["ranking_score"]["f1_mean"]
            rows = pd.read_csv(BASE_RES / out_name(b))
            tst = pd.read_csv(OUT / "out" / tag / b / "test_segs.csv")
            post_apply(rows, tst, *best[1], {s: v["dur"] for s, v in te.items()}).to_csv(res / out_name(b), index=False)
            info[name][b] = {"oof_base": base_sc, "oof_best": best[0], "params": best[1]}
            print(name, b, info[name][b], flush=True)
        (res / "choice.json").write_text(json.dumps(info[name], indent=1))


# ------------------------------------------------------------------------------------------- scoring
def score(variant, base=str(BASE_RES)):
    res = RES / variant if not Path(variant).is_dir() else Path(variant)
    preds = {k: res / f"fold_{k}.csv" for k in range(4)}
    A = {k: Path(base) / f"fold_{k}.csv" for k in range(4)}
    pooled = srcv.score_pooled(preds)
    pr = srcv.promote(A, preds, val_A=Path(base) / "val.csv", val_B=res / "val.csv")
    out = {"variant": variant, "pooled": summary(pooled), "promote_vs_norep": pr}
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"variant": variant, "pooled": out["pooled"],
                      **{k: pr[k] for k in ("delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved", "val_A",
                                            "val_B", "promote")},
                      "per_fold": [round(f["delta"], 4) for f in pr["per_fold"]]}, default=float), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "alone":
        alone(sys.argv[2], sys.argv[3:] or BUNDLES)
    elif cmd == "screen":
        screen(sys.argv[2], sys.argv[3:] or BUNDLES)
    elif cmd == "blend":
        blend(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:] or BUNDLES)
    elif cmd == "screenF":
        screen_final(sys.argv[2], sys.argv[3:] or BUNDLES)
    elif cmd == "post":
        post(sys.argv[2], sys.argv[3:] or BUNDLES)
    elif cmd == "score":
        score(*sys.argv[2:])

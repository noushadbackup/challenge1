"""Step 3: choose the boundary-refinement configuration on TRAIN OOF candidates only, then
apply the frozen configuration to val and score it once with the official evaluator CLI.
Selection uses the official scoring code in-process (abc2027.evaluate.evaluate); every reported
number is re-scored with the official CLI (--ignore-flagged).
Usage: python3 s3_select.py
"""
import itertools, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

PKG = Path("/home/user/data_raw/abc2027_uwb_train_v1")
sys.path.insert(0, str(PKG / "src"))
from abc2027.evaluate import evaluate, load_ground_truth, stream_durations, _match_stream  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
GT = load_ground_truth(PKG / "streams_events.csv", ignore_flagged=True)
DUR = {sp: stream_durations(PKG / "streams" / sp) for sp in ("train", "val")}


def tiou(a0, a1, b0, b1):
    i = np.maximum(0, np.minimum(a1, b1) - np.maximum(a0, b0))
    u = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(u > 0, i / np.where(u > 0, u, 1), 0.0)


def refine(D, cfg):
    """Return refined (t_start, t_end) arrays for the candidate table D."""
    s0, e0 = D.t_start.values.astype(float), D.t_end.values.astype(float)
    S = np.vstack([D[f"{c}_s"].values for c in cfg["crops"]]).astype(float)
    E = np.vstack([D[f"{c}_e"].values for c in cfg["crops"]]).astype(float)
    ok = np.isfinite(S) & np.isfinite(E) & (E > S) & (tiou(S, E, s0, e0) >= cfg["gate"])
    if cfg.get("noflag"):  # reject replays whose own review says start not visible
        for k, c in enumerate(cfg["crops"]):
            ok[k] &= ~D[f"{c}_rev"].fillna("").str.contains("start_not_visible").values
    Sm = np.where(ok, S, np.nan); Em = np.where(ok, E, np.nan)
    use = ok.sum(0) >= cfg.get("min_ok", 1)
    with np.errstate(all="ignore"):
        rs, re_ = np.nanmedian(Sm, 0), np.nanmedian(Em, 0)
    s = np.where(use, rs + cfg.get("ds", 0.0), s0 + cfg.get("pds", 0.0))
    e = np.where(use, re_ + cfg.get("de", 0.0), e0 + cfg.get("pde", 0.0))
    dmin = cfg.get("dmin", 0.0)
    e = np.maximum(e, s + dmin)  # duration floor: extend the end only (protects start error)
    return s, e, use


def to_preds(D, s, e, split):
    dur = DUR[split]
    out = []
    for sid, a, b, sc in zip(D.stream_id, s, e, D.score):
        a = max(0.0, a); b = min(dur[sid], b)
        if b > a:
            out.append({"stream_id": sid, "t_start": round(float(a), 3), "t_end": round(float(b), 3), "score": float(sc), "t_alarm": round(float(a), 3)})
    return out


def score(preds, split):
    r = evaluate(GT, preds, DUR[split])
    bt = r["by_theta"]
    return dict(rank=r["ranking_score"]["f1_mean"], f1=[bt[k]["f1"] for k in ("0.1", "0.3", "0.5", "0.7")],
                med_start=bt["0.3"]["start_error_median_abs"], tpfpfn=[bt["0.3"][k] for k in ("tp", "fp", "fn")])


def per_stream_counts(preds, split):
    """(n_streams, 3 thetas, [tp, fp, fn]) with the official matcher, for a stream-cluster bootstrap."""
    sids = sorted(DUR[split])
    C = np.zeros((len(sids), 3, 3))
    for i, sid in enumerate(sids):
        G = [g for g in GT if g["stream_id"] == sid]; Q = [p for p in preds if p["stream_id"] == sid]
        for k, th in enumerate((0.1, 0.3, 0.5)):
            r = _match_stream(G, Q, th)
            C[i, k] = [len(r["pairs"]), r["false_positive"], r["false_negative"]]
    return C


def boot_delta(Ca, Cb, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    def rank(C):
        t, f, m = C[..., 0], C[..., 1], C[..., 2]
        return np.mean(2 * t / np.maximum(2 * t + f + m, 1), axis=-1)
    idx = rng.integers(0, len(Ca), (n, len(Ca)))
    d = rank(Cb[idx].sum(1)) - rank(Ca[idx].sum(1))
    return float(np.mean(d)), [float(x) for x in np.percentile(d, [5, 95])], float(np.mean(d > 0))


def official_cli(preds, split, tag):
    p = OUT / f"{tag}.csv"; j = OUT / f"{tag}.json"
    pd.DataFrame(preds).to_csv(p, index=False)
    subprocess.run([sys.executable, "-m", "abc2027.evaluate", "--gt", "streams_events.csv", "--pred", str(p),
                    "--streams", f"streams/{split}", "--ignore-flagged", "--out", str(j)], cwd=PKG,
                   env={"PYTHONPATH": str(PKG / "src"), "PATH": "/usr/bin:/bin"}, check=True, capture_output=True)
    r = json.load(open(j)); bt = r["by_theta"]
    return dict(rank=round(r["ranking_score"]["f1_mean"], 4), f1=[round(bt[k]["f1"], 3) for k in ("0.1", "0.3", "0.5", "0.7")],
                tpfpfn=[bt["0.3"][k] for k in ("tp", "fp", "fn")], med_start=round(bt["0.3"]["start_error_median_abs"], 3),
                mean_tiou=round(bt["0.3"]["mean_tiou"], 3), fah=round(bt["0.3"]["false_alarms_per_hour"], 2))


def main():
    Dtr = pd.read_csv(OUT / "replay_train.csv")
    crops = sorted({c[:-2] for c in Dtr.columns if c.endswith("_s") and c not in ("t_s",)})
    base = to_preds(Dtr, Dtr.t_start.values, Dtr.t_end.values, "train")
    sb = score(base, "train")
    print("proto train", sb)
    log = []
    sets = [[c] for c in crops]
    sets += [["det2_5d", "det3_5d", "loc2_5d"], ["detq2_5d", "detq3_5d", "loc2_5d"], ["detq2_3d", "detq2_5d", "detq3_3d", "detq3_5d"],
             ["loc1.5_5d", "loc2_5d", "det2_5d", "detq2_5d"], ["detqr2_3d", "detqr2_5d", "detqr3_5d"], ["det1.5_5d", "det2_5d", "det3_5d"],
             ["loc2_3d", "loc2_5d", "det2_3d", "det2_5d", "detq2_3d", "detq2_5d"], ["det2_3d", "det2_5d", "det3_3d", "det3_5d"]]
    sets = [s for s in sets if all(c in crops for c in s)]
    for cs, g in itertools.product(sets, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
        for mo in ((1,) if len(cs) == 1 else (1, 2)):
            cfg = dict(crops=cs, gate=g, min_ok=mo)
            s, e, use = refine(Dtr, cfg)
            r = score(to_preds(Dtr, s, e, "train"), "train")
            log.append({**cfg, **r, "n_replaced": int(use.sum())})
    L = pd.DataFrame(log).sort_values(["rank", "med_start"], ascending=[False, True])
    L.to_csv(OUT / "select_stage1_train.csv", index=False)
    print(L.head(12).to_string())
    # stage 2: small grid of duration floor / end offsets on the top-5 crop+gate configs; start offsets fixed at 0
    # (start shifts trade tie-break 1 for tiny F1 changes; see README)
    log2 = []
    for _, row in L.head(5).iterrows():
        for dmin, de, pde in itertools.product((0.0, 1.0, 1.2), (0.0, 0.1), (0.0, 0.2)):
            cfg = dict(crops=row.crops, gate=row.gate, min_ok=row.min_ok, dmin=dmin, de=de, ds=0.0, pde=pde)
            s, e, use = refine(Dtr, cfg)
            r = score(to_preds(Dtr, s, e, "train"), "train")
            log2.append({**cfg, **r})
    L2 = pd.DataFrame(log2).sort_values(["rank", "med_start"], ascending=[False, True])
    L2.to_csv(OUT / "select_stage2_train.csv", index=False)
    print(L2.head(8).to_string())
    c1 = {k: L.iloc[0][k] for k in ("crops", "gate", "min_ok")}
    # stage 2 found no strict train gain over stage 1 (ties only), so candidate B is the stage-1 config plus the
    # structural duration floor of 1.0 s (evaluable GT never lasts < 1.0 s; MIN_EVENT_S in annotate.py), end-extension only
    c2 = {**c1, "dmin": 1.0}
    c1 = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in c1.items()}
    c2 = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in c2.items()}
    # promotion check on train OOF: stream-cluster bootstrap of the paired ranking-F1 delta
    Cp = per_stream_counts(base, "train")
    summary = {"A_replay": c1, "B_replay_post": c2, "train": {}, "val": {}, "boot_train": {}}
    Dva = pd.read_csv(OUT / "replay_val.csv")
    summary["train"]["proto"] = official_cli(base, "train", "train_proto_cli")
    summary["val"]["proto"] = official_cli(to_preds(Dva, Dva.t_start.values, Dva.t_end.values, "val"), "val", "val_proto_cli")
    for tag, cfg in (("A_replay", c1), ("B_replay_post", c2)):
        s, e, _ = refine(Dtr, cfg); P = to_preds(Dtr, s, e, "train")
        summary["train"][tag] = official_cli(P, "train", f"train_{tag}")
        summary["boot_train"][tag + "_vs_proto"] = boot_delta(Cp, per_stream_counts(P, "train"))
        s, e, use = refine(Dva, cfg)  # frozen config, val touched once per reported candidate
        summary["val"][tag] = official_cli(to_preds(Dva, s, e, "val"), "val", f"val_{tag}")
        summary["val"][tag]["n_replaced"] = int(use.sum())
    print(json.dumps(summary, indent=1, default=str))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

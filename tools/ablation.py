"""Ablation study of the submitted system (final_norep) on SR-CV: remove one component at a time.

    make ablation DEVICE=cuda               # 1 TCN bag (seed set 0)
    make ablation DEVICE=cuda BAGS=3        # 3 TCN bags (paper-grade; ~3x the time)
    make ablation-table                     # only rebuild the table from finished runs

Variants (each = configs/final_norep.json + one change):
  refit (HGB/TCN retrained on every fold, then predict):
    base        the submitted system
    no_fp       no background-fingerprint features         features.proto_set = "structure"
    physics     no structure (junction timing, segment) and no fingerprint features   proto_set = "physics"
    no_bands    no 48 range-band channels (TCN input = 64 prototype features)
    no_tcn      HGB only
    no_hgb      TCN only
  retune (the base bundles, post-model stages only):
    no_gate     no segment-classifier gate
    no_nms      no 4 s time NMS
    no_replay   no annotator replay (detector start/end)

Every run is resumable: a finished variant (summary.json + val_summary.json) is skipped.
Output: srcv/results/ablation/ablation.md and ablation.json (pooled SR-CV, 95% CI, delta vs base with the paired
subject-cluster bootstrap and folds improved, official val).
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import srcv  # noqa: E402
from srcv import core as C  # noqa: E402

BASE_CFG = json.loads((REPO / "uwbfall/configs/final_norep.json").read_text())
REFIT = {
    "base": {},
    "no_fp": {"features": {"proto_set": "structure"}},
    "physics": {"features": {"proto_set": "physics"}},
    "no_bands": {"features": {"use_bands": False}},
    "no_tcn": {"tcn": {"enabled": False}},
    "no_hgb": {"hgb": {"enabled": False}},
}
RETUNE = {
    "no_gate": {"clip": {"gate": "none"}},
    "no_nms": {"decode": {"one_event_per_clip": False}},
    "no_replay": {"boundary": {"method": "none"}},
}
LABEL = {
    "base": "submitted system (final_norep)", "no_fp": "- background fingerprint",
    "physics": "- structure and fingerprint features", "no_bands": "- range-band channels",
    "no_tcn": "- TCN (HGB only)", "no_hgb": "- HGB (TCN only)", "no_gate": "- segment gate",
    "no_nms": "- 4 s time NMS", "no_replay": "- annotator replay",
}
OUT = REPO / "srcv/results/ablation"
RES = REPO / "srcv/results"


def merge(a: dict, b: dict) -> dict:
    r = copy.deepcopy(a)
    for k, v in b.items():
        r[k] = merge(r[k], v) if isinstance(v, dict) and isinstance(r.get(k), dict) else copy.deepcopy(v)
    return r


def run_name(v: str, bag: int) -> str:
    return f"abl_{v}_b{bag}"


def done(name: str) -> bool:
    return (RES / name / "summary.json").exists() and (RES / name / "val_summary.json").exists()


def sh(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=REPO)


def refit(v: str, bag: int) -> None:
    name = run_name(v, bag)
    if done(name):
        print(f"[{name}] done, skipped", flush=True)
        return
    cfg = merge(merge(BASE_CFG, REFIT[v]), {"tcn": {"seed_offset": 10000 * bag}})
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(cfg, indent=1))
    if not (RES / name / "summary.json").exists():
        sh([sys.executable, "-m", "srcv.run_pipeline", name, "--config", str(p)])
    sh([sys.executable, "-m", "srcv.run_pipeline", name, "--config", str(p), "--val-only"])


def retune_variant(v: str, bag: int) -> None:
    import uwbfall
    from uwbfall.pipeline import retune

    name, base = run_name(v, bag), run_name("base", bag)
    if done(name):
        print(f"[{name}] done, skipped", flush=True)
        return
    out = RES / name
    out.mkdir(parents=True, exist_ok=True)
    runs = C.DATA / "runs"
    preds = {}
    for k in range(4):
        b = retune(runs / base / f"fold_{k}", srcv.pipeline_sources(k, "train"), C.events(k, "train"), RETUNE[v],
                   runs / name / f"fold_{k}")
        uwbfall.predict(b, srcv.pipeline_sources(k, "test")).to_csv(out / f"fold_{k}.csv", index=False)
        preds[k] = out / f"fold_{k}.csv"
        print(f"[{name}] fold {k} done", flush=True)
    from srcv import scoring as S
    res = {"name": name, "retune_of": base, "overrides": RETUNE[v],
           "pooled": S.summary(S.score_pooled(preds)), "bootstrap": S.bootstrap_ci(preds)}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    b = retune(runs / base / "official_train", uwbfall.split_sources("train"), C.real_events("train"), RETUNE[v],
               runs / name / "official_train")
    uwbfall.predict(b, uwbfall.split_sources("val")).to_csv(out / "val.csv", index=False)
    rep = S.summary(S.score_val(out / "val.csv"))
    (out / "val_summary.json").write_text(json.dumps({"val": rep}, indent=1, default=float))


def table(bags: int) -> None:
    from srcv import scoring as S

    variants = [v for v in [*REFIT, *RETUNE] if all(done(run_name(v, b)) for b in range(bags))]
    if "base" not in variants:
        print("no finished base run yet")
        return

    def preds(v):
        return [{k: RES / run_name(v, b) / f"fold_{k}.csv" for k in range(4)} for b in range(bags)]

    def mean(xs):
        return sum(xs) / len(xs)

    rows, base = [], preds("base")
    for v in variants:
        s = [json.loads((RES / run_name(v, b) / "summary.json").read_text())["pooled"] for b in range(bags)]
        val = mean([json.loads((RES / run_name(v, b) / "val_summary.json").read_text())["val"]["rank_f1"]
                    for b in range(bags)])
        ci = S.bootstrap_ci(preds(v))
        row = {"variant": v, "label": LABEL[v], "cv": mean([x["rank_f1"] for x in s]), "ci95": ci["ci95"],
               "f1@0.5": mean([x["f1@0.5"] for x in s]), "fp@0.3": mean([x["fp@0.3"] for x in s]),
               "start_med": mean([x["start_med@0.3"] for x in s]), "val": val}
        if v != "base":
            pr = S.promote(base, preds(v))  # B = the ablated system
            row.update({"delta": pr["delta"], "delta_ci95": pr["delta_ci95"],
                        "p_component_helps": 1.0 - pr["p_delta_gt_0"], "folds_worse": 4 - pr["n_folds_improved"],
                        "per_fold": pr["per_fold"]})
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps({"bags": bags, "rows": rows}, indent=1, default=float))
    lines = [f"# Ablation of final_norep on SR-CV ({bags} TCN bag{'s' if bags > 1 else ''}, 253 evaluable falls)", "",
             "Delta = ablated minus base (paired subject-cluster bootstrap, 5000 draws). P(helps) = probability that "
             "the removed component improves the score. Folds worse = folds (of 4) where removing it does not help.", "",
             "| Variant | SR-CV rank F1 [95% CI] | Delta [95% CI] | P(helps) | Folds worse | F1@0.5 | FP@0.3 | "
             "Median start err (s) | Val |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["variant"] == "base":
            d, ph, fw = "-", "-", "-"
        else:
            d = f"{r['delta']:+.3f} [{r['delta_ci95'][0]:+.3f}, {r['delta_ci95'][1]:+.3f}]"
            ph, fw = f"{r['p_component_helps']:.2f}", f"{r['folds_worse']}/4"
        lines.append(f"| {r['label']} | {r['cv']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {d} | {ph} | {fw} | "
                     f"{r['f1@0.5']:.3f} | {r['fp@0.3']:.1f} | {r['start_med']:.2f} | {r['val']:.3f} |")
    (OUT / "ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {OUT / 'ablation.md'}")


def main() -> None:
    a = sys.argv[1:]
    bags = int(a[a.index("--bags") + 1]) if "--bags" in a else 1
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    if "--table" not in a:
        if not (C.DATA / "fold_0").exists():
            sh([sys.executable, "-m", "srcv.build"])
        for b in range(bags):
            for v in REFIT:
                if only is None or v in only or v == "base":
                    refit(v, b)
            for v in RETUNE:
                if only is None or v in only:
                    retune_variant(v, b)
    table(bags)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()

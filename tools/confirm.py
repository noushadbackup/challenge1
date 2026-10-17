"""Confirmation run for the round-3 candidates on FRESH TCN seeds (never used for any selection).

    make confirm DEVICE=cuda            # round-3 suite: 3 fresh seed sets (TCN seed_offset 30000, 40000, 50000)
    make confirm DEVICE=cuda SUITE=r4   # round-4 suite: 9 TCN seeds, Chute04 segment features (seed sets 6, 7, 8)
    make confirm-table [SUITE=r4]       # only rebuild the table

Round-4 suite (SUITE=r4), each vs base on seed sets 6-8:
    seeds9        tcn.seeds = 9 (average 9 TCN members instead of 3: less seed noise)
    chute         clip.segclf.features = "phys+x" (extra segment features for weak falls such as Chute04)
    seeds9_chute  both
Suite r4x (SUITE=r4x): chute vs base on 6 seed sets (3-8); bags 3-5 re-use the r3 base runs, so only the
chute re-tunes of bags 3-5 run (about 30 min).

Candidates, each compared with the base on the same seeds (paired subject-cluster bootstrap):
    base       final_norep (the submitted system)
    no_bands   features.use_bands = false               (ablation: removing the 48 range-band channels helped)
    wdet       boundary.det_blend = true                (R3 "boundary": replay/detector boundary blend)
    nb_wdet    both
The promotion rule decides: P(delta > 0) >= 0.90, >= 3 of 4 folds improve, val >= base val - 0.02.
Resumable; output in srcv/results/confirm/confirm.md and confirm.json.
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
SUITES = {
    "r3": dict(bags=(3, 4, 5), prefix="cf", out="confirm",   # seed_offset = 10000 * bag; bags 0-2: the ablation
               refit={"base": {}, "no_bands": {"features": {"use_bands": False}}},
               retune={"wdet": ("base", {"boundary": {"det_blend": True}}),
                       "nb_wdet": ("no_bands", {"boundary": {"det_blend": True}})},
               label={"base": "final_norep (submitted)", "no_bands": "no range bands", "wdet": "boundary blend",
                      "nb_wdet": "no range bands + boundary blend"}),
    "r4": dict(bags=(6, 7, 8), prefix="c4", out="confirm_r4",
               refit={"base": {}, "seeds9": {"tcn": {"seeds": 9}}},
               retune={"chute": ("base", {"clip": {"segclf": {"features": "phys+x"}}}),
                       "seeds9_chute": ("seeds9", {"clip": {"segclf": {"features": "phys+x"}}})},
               label={"base": "final_norep (submitted)", "seeds9": "9 TCN seeds",
                      "chute": "Chute04 segment features", "seeds9_chute": "9 TCN seeds + Chute04 features"}),
    # r4x: the r4 winner on 6 seed sets. Bags 3-5 re-use the r3 base runs (cf_base_b*), bags 6-8 the r4 runs.
    "r4x": dict(bags=(3, 4, 5, 6, 7, 8), prefix="c4", out="confirm_r4x", base_prefix={3: "cf", 4: "cf", 5: "cf"},
                refit={"base": {}},
                retune={"chute": ("base", {"clip": {"segclf": {"features": "phys+x"}}})},
                label={"base": "final_norep (submitted)", "chute": "Chute04 segment features"}),
}
SUITE = "r3"
RES = REPO / "srcv/results"


def _suite(name_: str) -> None:
    global SUITE, BAGS, REFIT, RETUNE, LABEL, OUT
    SUITE = name_
    S_ = SUITES[name_]
    BAGS, REFIT, RETUNE, LABEL, OUT = S_["bags"], S_["refit"], S_["retune"], S_["label"], RES / S_["out"]


_suite("r3")


def merge(a: dict, b: dict) -> dict:
    r = copy.deepcopy(a)
    for k, v in b.items():
        r[k] = merge(r[k], v) if isinstance(v, dict) and isinstance(r.get(k), dict) else copy.deepcopy(v)
    return r


def name(v: str, bag: int) -> str:
    S_ = SUITES[SUITE]
    pre = S_.get("base_prefix", {}).get(bag, S_["prefix"]) if v == "base" else S_["prefix"]
    return f"{pre}_{v}_b{bag}"


def done(n: str) -> bool:
    return (RES / n / "summary.json").exists() and (RES / n / "val_summary.json").exists()


def sh(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=REPO)


def refit(v: str, bag: int) -> None:
    n = name(v, bag)
    if done(n):
        print(f"[{n}] done, skipped", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{n}.json"
    p.write_text(json.dumps(merge(merge(BASE_CFG, REFIT[v]), {"tcn": {"seed_offset": 10000 * bag}}), indent=1))
    if not (RES / n / "summary.json").exists():
        sh([sys.executable, "-m", "srcv.run_pipeline", n, "--config", str(p)])
    sh([sys.executable, "-m", "srcv.run_pipeline", n, "--config", str(p), "--val-only"])


def retune_variant(v: str, bag: int) -> None:
    import uwbfall
    from srcv import scoring as S
    from uwbfall.pipeline import retune

    parent, ov = RETUNE[v]
    n, pn = name(v, bag), name(parent, bag)
    if done(n):
        print(f"[{n}] done, skipped", flush=True)
        return
    out = RES / n
    out.mkdir(parents=True, exist_ok=True)
    runs = C.DATA / "runs"
    preds = {}
    for k in range(4):
        b = retune(runs / pn / f"fold_{k}", srcv.pipeline_sources(k, "train"), C.events(k, "train"), ov,
                   runs / n / f"fold_{k}")
        uwbfall.predict(b, srcv.pipeline_sources(k, "test")).to_csv(out / f"fold_{k}.csv", index=False)
        preds[k] = out / f"fold_{k}.csv"
        print(f"[{n}] fold {k} done", flush=True)
    (out / "summary.json").write_text(json.dumps(
        {"name": n, "retune_of": pn, "overrides": ov, "pooled": S.summary(S.score_pooled(preds)),
         "bootstrap": S.bootstrap_ci(preds)}, indent=1, default=float))
    b = retune(runs / pn / "official_train", uwbfall.split_sources("train"), C.real_events("train"), ov,
               runs / n / "official_train")
    uwbfall.predict(b, uwbfall.split_sources("val")).to_csv(out / "val.csv", index=False)
    (out / "val_summary.json").write_text(json.dumps({"val": S.summary(S.score_val(out / "val.csv"))}, indent=1,
                                                     default=float))


def table() -> None:
    from srcv import scoring as S

    variants = [v for v in [*REFIT, *RETUNE] if all(done(name(v, b)) for b in BAGS)]
    if "base" not in variants:
        print("no finished base run yet")
        return

    def preds(v):
        return [{k: RES / name(v, b) / f"fold_{k}.csv" for k in range(4)} for b in BAGS]

    def mean(xs):
        return sum(xs) / len(xs)

    def val(v):
        return mean([json.loads((RES / name(v, b) / "val_summary.json").read_text())["val"]["rank_f1"] for b in BAGS])

    rows = []
    for v in variants:
        s = [json.loads((RES / name(v, b) / "summary.json").read_text())["pooled"] for b in BAGS]
        row = {"variant": v, "label": LABEL[v], "cv": mean([x["rank_f1"] for x in s]),
               "f1@0.5": mean([x["f1@0.5"] for x in s]), "fp@0.3": mean([x["fp@0.3"] for x in s]),
               "start_med": mean([x["start_med@0.3"] for x in s]), "val": val(v)}
        if v != "base":
            pr = S.promote(preds("base"), preds(v), val_A=val("base"), val_B=val(v))
            row.update({k: pr[k] for k in ("delta", "delta_ci95", "p_delta_gt_0", "n_folds_improved", "per_fold",
                                           "val_ok", "promote")})
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{OUT.name}.json").write_text(json.dumps({"bags": BAGS, "rows": rows}, indent=1, default=float))
    lines = [f"# Confirmation, suite {SUITE}, on fresh TCN seeds (bags {', '.join(map(str, BAGS))}; SR-CV, 253 evaluable falls)", "",
             "Each candidate vs the base on the same seeds. Rule: P >= 0.90, >= 3/4 folds improve, val >= base - 0.02.",
             "", "| Variant | SR-CV rank F1 | Delta [95% CI] | P(gain) | Folds improved | Val | F1@0.5 | FP@0.3 | "
             "Median start err (s) | Passes |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["variant"] == "base":
            d, p, f, ok = "-", "-", "-", "-"
        else:
            d = f"{r['delta']:+.4f} [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}]"
            p, f, ok = f"{r['p_delta_gt_0']:.3f}", f"{r['n_folds_improved']}/4", "yes" if r["promote"] else "no"
        lines.append(f"| {r['label']} | {r['cv']:.4f} | {d} | {p} | {f} | {r['val']:.4f} | {r['f1@0.5']:.3f} | "
                     f"{r['fp@0.3']:.1f} | {r['start_med']:.2f} | {ok} |")
    (OUT / f"{OUT.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {OUT / (OUT.name + '.md')}")


def main() -> None:
    a = sys.argv[1:]
    _suite(a[a.index("--suite") + 1] if "--suite" in a else "r3")
    if "--table" not in a:
        from _preflight import check
        check()
        if not (C.DATA / "fold_0").exists():
            sh([sys.executable, "-m", "srcv.build"])
        for b in BAGS:
            for v in REFIT:
                refit(v, b)
            for v in RETUNE:
                retune_variant(v, b)
    table()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()

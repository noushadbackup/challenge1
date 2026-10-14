"""Re-score saved SR-CV runs with the official evaluator and check them against their stored summary.json.

    python3 -m srcv.rescore [name ...]          # default: every srcv/results/<name>/ holding fold_0..3.csv

For each run: per-fold official ranking F1 / TP-FP-FN, pooled OOF report, subject-cluster bootstrap CI,
pooled F1 without the Test_P3_3 cluster, and (if val.csv exists) the official val score. Writes
srcv/results/rescore.json and reports whether every stored number was reproduced.
"""
from __future__ import annotations

import json
import sys

from . import core as C
from . import scoring as S
from .compare import _arm


def rescore(name: str) -> dict:
    d = C.HERE / "results" / name
    preds = _arm(str(d))
    per_fold = {k: S.summary(S.score(p, k)) for k, p in preds.items()}
    pooled = S.summary(S.score_pooled(preds))
    out = {"per_fold": per_fold, "pooled": pooled, "bootstrap": S.bootstrap_ci(preds),
           "pooled_without_Test_P3_3": S.rank_f1_by_subject_mask(preds, ["Test_P3_3"]),
           "fold_mean": sum(v["rank_f1"] for v in per_fold.values()) / 4}
    if (d / "val.csv").exists():
        out["val"] = S.summary(S.score_val(d / "val.csv"))
    stored = next((p for p in (d / "summary.json", d / "summary_full.json") if p.exists()), d / "summary.json")
    if stored.exists():
        st = json.loads(stored.read_text())
        ok = abs(st["pooled"]["rank_f1"] - pooled["rank_f1"]) < 1e-12 and all(
            abs(r["test"]["rank_f1"] - per_fold[r["fold"]]["rank_f1"]) < 1e-12 for r in st["per_fold"])
        out["matches_stored_summary"] = bool(ok)
    return out


def main() -> None:
    names = sys.argv[1:] or sorted(p.name for p in (C.HERE / "results").iterdir()
                                   if p.is_dir() and all(any(p.glob(f"fold_{k}*.csv")) for k in range(4)))
    res = {}
    for n in names:
        res[n] = r = rescore(n)
        pf = " ".join(f"{r['per_fold'][k]['rank_f1']:.4f}" for k in range(4))
        print(f"{n:28s} pooled {r['pooled']['rank_f1']:.4f} CI [{r['bootstrap']['ci95'][0]:.3f}, "
              f"{r['bootstrap']['ci95'][1]:.3f}] folds {pf} noP3_3 {r['pooled_without_Test_P3_3']:.4f} "
              f"tp/fp/fn@0.3 {r['pooled']['tp@0.3']}/{r['pooled']['fp@0.3']}/{r['pooled']['fn@0.3']} "
              f"val {r.get('val', {}).get('rank_f1', float('nan')):.4f} stored-match {r.get('matches_stored_summary')}",
              flush=True)
    out = C.HERE / "results" / "rescore.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(res)
    out.write_text(json.dumps(old, indent=1, default=float))


if __name__ == "__main__":
    main()

# I3 — event hit model (score column) and operating point (PLAN v2 W5, expert X4)

Report: `reports/round2/I3.md`. Nothing here is promoted; the default config (G7N4) is unchanged.

Environment:

    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src

All variants start from the reference G7N4 (I1): same detector, same operating point, same replay, same 4 s time
NMS. They change only the score column, the cut threshold, the NMS order, or the operating point. Scoring is done
with the official evaluator (SR-CV over 3 TCN bags; val through the official CLI) via `score.py`, which wraps
`work/R2_EVAL/score_all.py`.

## Scripts (run order)

| script | what it does |
|---|---|
| `run.py <targets>` | Hit model trained on each target's **training OOF** candidates (inside the fit, like the operating point). Writes REF (byte-identical to `work/I1_clip/preds/<bag>/G7N4`, asserted), T0, H, HT, HN, HTN, plus `cache/feat_<bag>_<t>.pkl` (test-part features/rows) and `out/info_<t>.json`. |
| `run2.py [--pool-bags] [--fixed-tau 0.46] [--nms-det] --tag X` | Hit model trained on **SR-CV OOF** candidates (the held-out test parts of the other 3 folds; all 4 for val). Writes XT0/XH/XHT + tag, `out/info_X<tag>.json`. |
| `run3.py <targets>` | Recall-oriented candidate set (thr→0.2, min_dur→0.4) with replays → `cache/featR_*` (for `run2.py --feat featR`, "R" variants). |
| `opgrid.py <targets>` | Replays every decode-grid candidate of the SR-CV test parts and val → `cache/repG_<t>_test.pkl`. |
| `opsel.py` | Operating point chosen on SR-CV OOF (leave-target-fold-out): XOPd (detector-level F1, as `decode.tune`) and XOPf (final F1 after replay + NMS). It asserts that the reference op reproduces REF. → `out/opsel.json`. |
| `fit_hit.py <out.pkl> [--fixed-tau 0.46]` | Deployable hit model (all 4 SR-CV folds, pooled bags) for `python -m uwbfall hitattach`. `/home/user/data_raw/I3/hit_XHTPF.pkl` was made with `--fixed-tau 0.46`. |
| `ign.py <variants>` | Ignore-zone audit → `out/ignore_audit.json`. |
| `taucurve.py` | Diagnostic only: held-out F1 against tau (hindsight, not used for selection) → `out/taucurve.json`. |
| `score.py table V1,V2` / `score.py promote REF V` | Scoring and promotion rule → `out/table.json`, `out/promote_REF_vs_V.json`. |
| `mdtable.py V1,V2,...` | Markdown tables for the report. |

## Variant names

- `T` means a threshold cut after NMS. `H` means the hit-model score. `X` means the model and tau were learned on SR-CV OOF test parts instead of training OOF.
- `P` means the hit model is trained on the rows of all 3 bags together. `F` means a fixed tau of 0.46 (the F1*/2 rule) instead of the nested choice.
- `N` means the NMS is ordered by the detector score, so the kept rows are the reference's and the hit score acts only through the cut.
- `R` means the recall-oriented candidate set. `OPd`/`OPf` mean an operating point re-selected on SR-CV OOF.

## Package hooks (uwbfall, all off by default)

- `uwbfall/hitmodel.py`: features, labels (free ignore-zone rows are excluded from training), HitModel, nested tau.
- Config `score.hit_model` (null | "lgbm"), `score.threshold` (null | tau), `score.nms_order` ("score" | "detector").
- `python -m uwbfall hitattach --models <bundle> --hit <hit.pkl> [--tau 0.46] [--nms-order detector] --out <dir>`.
- Parity: `python3 -m uwbfall.tests.parity_hit` → ALL OK (`logs/parity_hit.log`). It checks three things:
  - hit model off reproduces G7N4 val;
  - XHTPF val is reproduced byte-for-byte;
  - XHTPFN val (`nms_order=detector`) is reproduced byte-for-byte.

Caches are in `cache/` (git-ignored). The hit-model pickles and parity bundles are in `/home/user/data_raw/I3/`.

# I1: clip-level decoding, segment classifier, post-fall class (PLAN v2 W3, experts X2/X3)

Report: `reports/round2/I1.md`. Library code: `uwbfall/clip.py`, plus the clip stage in `uwbfall/pipeline.py`
(`_fit_clip`, `_nms`, `stream_probs`, `retune`).

Environment for every command:

    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src

No TCN was retrained here. Every variant reuses the fitted SR-CV bundles of the three R2_EVAL bags in
`/home/user/data_raw/srcv/runs/{uwb_default,tcn_bag1,tcn_bag2}/{fold_k,official_train}`: HGB, PCA, TCN bag, OOF
probabilities and the cached test probabilities `work/R2_EVAL/cache/probs_<bag>_<t>.npz`. Each variant re-tunes its
operating point on its own training OOF, using the official evaluator, the uwbfall grid and the first-max rule. It
then runs the reference annotator replay (gate 0.3).

## Pipeline (order of execution)

| step | script | output |
|---|---|---|
| 1. junction statistic + threshold (fall-junction recall 0.95 on the train streams), segment tables, 3-class HGB | `prep.py <t> --hgb3` (`run_prep.sh`, about 5 min per target) | `cache/prep_<t>.pkl`, `cache/hgb3_<t>.{npz,pkl}`, `logs/prep_{a,b}.log` |
| 2. segment-classifier feature set, chosen on subject-grouped train OOF only | `segclf_ablate.py` | `out/segclf_ablate*.json` (nofp wins: mean AUC 0.975, log-loss 0.240) |
| 3. per-block gates: subject-grouped OOF on train, final model on test | `segclf_final.py nofp gate` | key `gate` in `cache/prep_<t>.pkl` |
| 4. gate form and floor, chosen on train OOF only | `gate_oof.py <bag>` | `out/gate_oof_<bag>.json` |
| 5. NMS gap, chosen on train OOF only (the bundles' `oof_final.csv`) | `nms_oof.py` | `out/nms_oof_uwb_default.json` |
| 6. decode variants per (bag, target) | `variants.py <bag> <targets> <V> [--gate-floor f --suffix s]` (`run_D.sh`, `run_G*.sh`) | `preds/<bag>/<V>/{fold_k,val}.csv`, `cache/ops_*.json` |
| 7. post-hoc NMS | `apply_nms.py <src> <dst> time:4 \| segment` | `preds/<bag>/<dst>/` |
| 8. scoring, using the R2_EVAL tooling (`srcv.promote`; val via the official CLI) | `score.py table <V,...> <bags>`, `score.py promote A B <bags>` | `out/table.json`, `out/promote_A_vs_B.json`, `logs/table.log` |
| 9. error anatomy @0.3 | `erran.py <pred_dirs> 0.3` | `out/erran_0.3.txt` |

Variants: `REF` is the R2_EVAL C3G3 reference (symlink; `REFx` is the byte-identical rebuild of fold 0 by
variants.py). `N<g>` applies time NMS with gap g s to REF. `NS` applies segment NMS to REF. `G` gates by
P(fall clip) with floor 0, `G5` with floor 0.5 and `G7` with floor 0.7. `D` replaces the HGB P(fall) with the
3-class HGB. Combinations are named by concatenation (`G7N4` = G7, then time NMS with a 4 s gap).

## Integration checks

- `python3 -m uwbfall.tests.parity_clip_retune official,0` runs `pipeline.retune` (fit's post-model code path) and
  `pipeline.predict` with the promoted config on the bag-0 bundles. It checks that the operating point and the
  val / fold-0 CSVs equal `preds/uwb_default/G7N4/`. Log: `logs/parity_clip_retune.log`.
- `python3 uwbfall/tests/smoke_clip.py` checks time NMS and gate=mult (floor 0) against N4/G.
- `python3 uwbfall/tests/smoke_clip_fit.py` runs an end-to-end `fit()` + `predict()` with the clip stage
  (HGB only). Log: `logs/smoke_clip_fit.log`.
- `python3 -m uwbfall.tests.check_configs` checks that reference.json equals DEFAULT_CONFIG and that the pinned
  configs differ from it only where intended.

Caches (`cache/`, about 41 MB) are git-ignored. Retune parity bundles are in `/home/user/data_raw/I1/`.

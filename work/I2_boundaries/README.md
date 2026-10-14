# I2: boundaries (Chute04), D3 hedge, ignore-zone loss masking (PLAN v2 W4; experts X1 / X3)

Report: `reports/round2/I2.md`. Library code is in `uwbfall`, behind config flags that are **off by default**:
- `features.zones` / `features.zones_for` (features.py `zone_features`, `hgb_design`)
- `labels.ignore_clips` / `labels.ignore_members` (features.py `ignore_clip_mask`; models.py `fit_hgb(w)`, `seqs(w)`, `_wloss`)
- `boundary.agg` / `boundary.cluster_iou`, `boundary.fallback` / `boundary.contain_frac` (boundary.py `refine`, `_largest_cluster`)
- `boundary.selector` (boundary.py `hypotheses`, `selector_targets`, `fit_selector`, `select`). This one is experiment-only, and `check_implemented` refuses it in fit/predict.

Environment for every command:

    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src

Every variant starts from the current reference, **G7N4** (I1). It reuses the fitted SR-CV bundles of the 3 R2_EVAL bags in `/home/user/data_raw/srcv/runs/{uwb_default,tcn_bag1,tcn_bag2}`, the R2_EVAL test probabilities, and the I1 segment gates (`work/I1_clip/cache/prep_<t>.pkl`). Each variant re-tunes its operating point on its own G7-gated train OOF, using the official evaluator, the uwbfall grid and the first-max rule. It then runs the reference replay (gate 0.3) and time NMS at 4 s.

`REF` in `preds/<bag>/` is a symlink to `work/I1_clip/preds/<bag>/G7N4`. `decvar.py --check` with no option rebuilds it byte-for-byte.

## Pipeline

| step | script | output |
|---|---|---|
| 1. Replay internals of the G7 candidates, train OOF and test, union of the 3 bags. Byte-checked against I1 G7N4 for every (bag, target). | `dump.py <targets> --parts test,train` (`run_dump.sh`) | `cache/rep_<t>_{train,test}.pkl` |
| 2. Per-type matched rates and Test_P3_3 recall | `pertype.py <pred root> <variant>` | `out/pertype.txt` |
| 3. Failure modes per fall type (det_miss / replay_worse / both_bad / fallback / nms_drop) | `erb.py <theta>` | `out/erb_events_<theta>.csv`, `out/erb_0.5.txt` |
| 4. Train-only diagnostics for the crop-aggregation rules (cluster, containment fallback) | `aggdiag.py <targets>` | `out/aggdiag.json` |
| 5. (a) Boundary-hypothesis selector, trained on train-OOF replay internals; margin chosen by stream-grouped inner CV on train | `sel.py <targets>` | `preds/<bag>/S/`, `out/sel_S.json` |
| 6. (b)/(c) HGB member refits: zones Z (3 zones), Z0 (R29171 bins 40–68 only), W (ignore-clip zero weight), Nul (control: max_iter 290). `base` reproduces the bundle HGB exactly. | `hgbvar.py <targets> <variants>` (`run_hgb.sh`, `run_nul.sh`) | `cache/hgb_<V>_<t>.npz` |
| 7. (c) TCN arm: TCN-only bag with zero-weight ignore clips (bag-0 seeds, a 1-bag screen) | `tcnw.py tcnW_bag0 0 <targets>` (`run_tcnw_a.sh`) | `/home/user/data_raw/I2/runs/tcnW_bag0/`, `cache/probs_tcnW_bag0_<t>.npz` |
| 8. Decode variants: re-tuned operating point, replay (row cache), refine options, NMS | `decvar.py <targets> <name> [--hgb V] [--tcn bag] [--agg cluster] [--fallback contain] [--bags ..] [--check]` | `preds/<bag>/<name>/{fold_k,val}.csv`, `out/ops_<name>.json` |
| 9. Scoring and promotion rule (R2_EVAL tooling: `srcv.promote`, 5000 subject-cluster resamples; val via the official CLI) | `score.py table V,..`, `score.py promote REF V [bags]`, `mdtable.py V,..` | `out/table.json`, `out/promote_REF_vs_<V>.json` |

## Integration checks

`python3 -m uwbfall.tests.parity_i2 Z,W` (log: `logs/parity_i2.log`) checks four things:
- `fit()` with `features.zones` or `labels.ignore_clips=zero_weight` reproduces the experiment's OOF and val HGB probabilities bit-for-bit, through `stream_probs`.
- The default config reproduces the uwb_default bundle's HGB (`base`).
- The TCN weighted loss equals the unweighted loss for unit weights, and gives zero loss and zero gradient for zero weights.
- The cluster and contain rules reduce to the median on a single-mode candidate.

`python3 -m uwbfall.tests.check_configs` still passes: `reference.json` equals DEFAULT_CONFIG, because nothing was promoted.

Caches in `cache/` are git-ignored (`work/*/cache/`). TCN-W bundles are outside git in `/home/user/data_raw/I2/`.

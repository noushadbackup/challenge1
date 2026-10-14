# E5: ensemble and multi-agent voting

## What this is
This package builds the ensemble layer from PLAN §3 "Multi-agent / voting" and compares every voting scheme under one protocol:

- **(a) Radar agents.** One LightGBM per radar, plus a fused model.
- **(b) Model committee.** LightGBM, XGBoost, CatBoost and HistGB, each bagged over folds x seeds.
- **(c) Logistic stacking.**
- **(d) Event-level 1-D weighted segment fusion (WBF) and soft-NMS.**
- **(e) A physics-only hedge member.**

As an extra committee member it also reuses E4's TCN probabilities: the OOF and val outputs of `tcn_all`, copied to `cache/probs_tcn_e4.npz`.

## Protocol
- **Features.** A snapshot of E4's block cache (`cache/blocks_e4.npz`, sha256 `ad3ac808…`). It holds the 64 prototype "full" features at 0.2 s, built by the same code as `analysis/prototype_detector.py`, plus 48 range-band channels. `e5lib.frame()` adds per-radar ±1/±2 s lags.
- **Folds.** The prototype's leave-streams-out `GroupKFold(4)` on the 13 train streams.
- **Bagging.** Each member is trained as 4 folds x 2 seeds = 8 fits.
  - OOF = the seed-mean of the held-out fold models.
  - Val = the mean of all 8 fold models. There is no refit, so the OOF-tuned threshold transfers.
- **Determinism.** HGB uses `early_stopping=False`, which removes the hidden random 10% split (X5). LightGBM uses `deterministic=True`. Every run uses n_jobs=2. Peak RSS was ≤1.04 GB.
- **Selection.** Every scheme's knobs were chosen on **train OOF only**, using the official `abc2027.evaluate` code in-process (durations from the npz, verified identical by E1). The knobs are:
  - the prototype post-processing grid (thr, smooth, min_dur, gap)
  - the WBF min_support and min_score
  - the soft-NMS threshold
  - the TCN weight
  - the stacking meta-model, cross-fitted on the same folds
- **Val scoring.** Each candidate was then scored **once** on val with the **official CLI** (`--ignore-flagged`, h5 streams). No val number fed back into any choice.
- **Uncertainty.** `bootstrap.py` runs a paired segment-cluster bootstrap with 5000 resamples of composition segments, using official `_match_stream` counts. Its point estimates equal the CLI values exactly.

## Results (official evaluator; val = 5 streams, 67 evaluable falls)
The reference is the prototype "full" at val 0.8840. `single:hgb_proto` is the same model, made deterministic and fold-bagged: 0.8822.

| scheme | train OOF rank F1 | **val rank F1** | val F1@.1/.3/.5/.7 | TP/FP/FN @.3 | med \|start err\| |
|---|---|---|---|---|---|
| committee+TCN, w_tcn=0.75 (**OOF winner overall**) | **0.9464** | **0.9141** | .970/.955/.818/.591 | 63/2/4 | 0.34 |
| single TCN (E4 output, reused) | 0.9404 | 0.9303 | .970/.970/.851/.507 | 65/2/2 | 0.34 |
| committee+TCN, w_tcn=0.5 | 0.9409 | 0.8790 | .948/.919/.770/.504 | 62/6/5 | 0.34 |
| WBF-1D committee+TCN (support 5/5) | 0.9389 | 0.8872 | .969/.923/.769/.477 | 60/3/7 | 0.33 |
| soft-NMS committee+TCN | 0.9348 | 0.9026 | .954/.954/.800/.492 | 62/1/5 | 0.32 |
| stacking (all members + TCN) | 0.9280 | 0.8790 | .948/.919/.770/.489 | 62/6/5 | 0.34 |
| WBF-1D committee (support ≥1, score ≥0.4) (**best GBDT-only on OOF**) | 0.9203 | 0.8753 | .947/.916/.763/.458 | 60/4/7 | 0.34 |
| radar: mean(fused, max radar agent) | 0.9180 | 0.8622 | .947/.887/.752/.466 | 59/7/8 | 0.32 |
| hedge: committee + 0.2·physics | 0.9173 | 0.8753 | .947/.901/.779/.519 | 59/5/8 | 0.28 |
| **committee soft vote (mean of 4 GBDT)** | 0.9162 | 0.8804 | .947/.901/.794/.489 | 59/5/8 | 0.28 |
| committee logit-mean | 0.9162 | 0.8804 | identical metrics | 59/5/8 | 0.28 |
| soft-NMS committee | 0.9151 | 0.8855 | .947/.931/.779/.443 | 61/3/6 | 0.34 |
| single XGBoost (fused 112 f.) | 0.9135 | 0.8636 | .939/.894/.758/.485 | 59/6/8 | 0.26 |
| radar: fused OR ≥2-of-3 agents agree | 0.9121 | 0.8636 | .939/.894/.758/.515 | 59/6/8 | 0.28 |
| hedge: committee + 0.33·physics | 0.9119 | 0.8702 | .947/.901/.763/.519 | 59/5/8 | 0.28 |
| single LightGBM (fused) | 0.9107 | 0.8657 | .955/.896/.746/.418 | 60/7/7 | 0.32 |
| stacking (committee + radars + physics) | 0.9096 | 0.8706 | .940/.896/.776/.418 | 60/7/7 | 0.40 |
| stacking (committee) | 0.9070 | 0.8622 | .932/.887/.767/.496 | 59/7/8 | 0.26 |
| soft-NMS committee+radars | 0.9054 | 0.8360 | .905/.889/.714/.397 | 56/3/11 | 0.34 |
| single CatBoost (fused) | 0.9036 | 0.8346 | .919/.889/.696/.415 | 60/8/7 | 0.30 |
| radar: mean(fused, 3 agents) | 0.9027 | 0.8557 | .955/.896/.716/.448 | 60/7/7 | 0.31 |
| single HGB proto (deterministic, bagged) | 0.9016 | 0.8822 | .947/.902/.797/.541 | 60/6/7 | 0.24 |
| single HGB (fused) | 0.9008 | 0.8722 | .947/.917/.752/.451 | 61/5/6 | 0.34 |
| committee rank-mean (quantile grid) | 0.8960 | 0.8657 | .940/.881/.776/.522 | 59/8/8 | 0.28 |
| physics-only hedge alone (LightGBM) | 0.8781 | 0.8307 | .905/.857/.730/.444 | 54/5/13 | 0.32 |
| radar: mean of 3 agents | 0.8776 | 0.8448 | .931/.885/.718/.473 | 58/6/9 | 0.26 |
| radar: **≥2-of-3 agents agree** | 0.8624 | 0.8462 | .923/.877/.738/.462 | 57/6/10 | 0.34 |
| radar: max agent | 0.8527 | 0.8198 | .904/.874/.681/.459 | 59/9/8 | 0.26 |
| radar agent R29444 / R30722 / R29171 alone | 0.821 / 0.820 / 0.777 | 0.786 / 0.825 / 0.769 | | | |

Full numbers are in `out/combine_results.json` and `logs/combine.log`. The scored CSVs are `out/val_*.csv` with their official JSONs in `out/val_*.json`, and the OOF events are in `out/oof_*.csv`.

### Paired segment bootstrap (Δ rank F1, 95% CI, P(Δ>0))
| comparison | train OOF | val |
|---|---|---|
| committee mean − HGB proto | +0.015 [−0.004, +0.033], 0.94 | −0.002 [−0.046, +0.043], 0.45 |
| WBF committee − committee mean | +0.004 [−0.007, +0.016], 0.75 | −0.005 [−0.023, +0.011], 0.20 |
| soft-NMS committee − committee mean | −0.001 [−0.017, +0.015], 0.44 | +0.005 [−0.011, +0.023], 0.62 |
| stacking (committee) − committee mean | −0.009 [−0.023, +0.005], 0.09 | −0.018 [−0.047, +0.009], 0.09 |
| hedge w0.2 − committee mean | +0.001 [−0.010, +0.013], 0.55 | −0.005 [−0.017, 0.000], 0.00 |
| ≥2-of-3 radars agree − fused LightGBM | **−0.048 [−0.081, −0.019], 0.00** | −0.020 [−0.077, +0.034], 0.23 |
| mean of 3 radar agents − fused | −0.033 [−0.065, −0.002], 0.02 | −0.021 [−0.065, +0.019], 0.16 |
| fused OR 2-of-3 − fused | +0.001 [−0.020, +0.023], 0.55 | −0.002 [−0.039, +0.035], 0.45 |
| committee+TCN w0.75 − HGB proto | **+0.045 [+0.021, +0.072], 1.00** | +0.032 [−0.011, +0.078], 0.92 |
| committee+TCN w0.75 − TCN alone | +0.006 [−0.004, +0.017], 0.88 | −0.016 [−0.060, +0.025], 0.22 |
| WBF committee+TCN − TCN alone | −0.002 [−0.022, +0.019], 0.44 | −0.043 [−0.088, +0.001], 0.03 |

## Verdict: which voting scheme wins
1. **Overall winner on train OOF: a frame-level weighted soft vote of 0.75·TCN + 0.25·mean(4 GBDT).**
   - It scores OOF 0.9464 and **val 0.9141**, which is +0.030 over the 0.884 reference. Against the bagged HGB it is +0.032, CI [−0.011, +0.078], P=0.92.
   - However, the TCN alone scores higher on val (0.9303), and on val the blend is not better than the TCN (−0.016, CI [−0.060, +0.025]).
   - The +0.006 OOF edge over the TCN is within noise. The TCN carries the gain; the GBDT committee adds nothing measurable on top of it.
2. **Among GBDT-only schemes, the plain equal-weight soft vote (probability mean) is the scheme to keep.**
   - It beats every single GBDT on OOF: +0.015 over the HGB prototype (P=0.94), and +0.003 to +0.015 over the individual LightGBM, XGBoost, CatBoost and HGB members.
   - On val it is level with the reference (0.8804 vs 0.8822). It has lower variance and needs no learned knobs.
   - Logit-mean is identical. Rank-mean is worse (0.896 OOF).
3. **The "≥2 radar agents agree" rule is clearly harmful.**
   - On OOF it costs −0.048, CI excluding 0. On val it reaches 0.846 vs 0.866 for the fused model.
   - Single-radar agents are weak (0.78-0.82 OOF), because each mat is seen by one radar (F5), as X1 and X3 predicted.
   - "Fused OR 2-of-3" and "mean(fused, max agent)" are neutral. **Drop radar voting; use one fused model.** Anchor-radar features are E3's or E2's job.
4. **Logistic stacking does not help.** It is −0.009 OOF and −0.018 val against the simple mean. With about 250 OOF events and correlated members, the meta-model overfits (see the sign-flipping coefficients in `combine_results.json`).
5. **Event-level WBF-1D and soft-NMS are neutral at best.**
   - WBF: +0.004 OOF and −0.005 val against the frame mean, and −0.043 val when the TCN is in the pool.
   - Soft-NMS: −0.001 OOF and +0.005 val.
   - Under F1, score order does not matter (X2), so these only add knobs. Fuse at frame level and decode once.
6. **The physics-only hedge** scores 0.878 OOF and 0.831 val alone (the prototype's physics set gave 0.790 val, but this hedge also has range bands). As a 0.2 blend member it is neutral on OOF (+0.001) and costs −0.005 on val. **Keep it as a switch, not a blend member** (agrees with X4).

## How to run
```
cd /home/user/challenge1/work/E5_ensemble
OMP_NUM_THREADS=2 python3 members.py hgb_proto lgb_fused xgb_fused cat_fused hgb_fused lgb_radar0 lgb_radar1 lgb_radar2 lgb_physics   # ~10 min, ≤1.1 GB
python3 combine.py                     # ~90 s: tunes on OOF, scores each candidate once on val (official CLI)
python3 bootstrap.py train out/oof_single__hgb_proto.csv out/oof_committee__mean.csv ...   # paired CIs
python3 bootstrap.py val   out/val_single__hgb_proto.csv out/val_committee__mean.csv ...
```
`cache/blocks_e4.npz` and `cache/probs_tcn_e4.npz` are copies of E4's files (`work/E4_deep_model/cache/blocks.npz`, `out/probs_tcn_all.npz`). Rebuilding them needs E4's `data.py` and `tcn.py`.

## Honest limitations
- **Train OOF is leaky**: stream-grouped folds share ADL files and subjects (data verdict). The post-processing is also chosen on the same OOF it is scored on. The train-OOF CIs are therefore optimistic. They are only good for relative comparisons, and even those may be biased toward high-capacity members.
- **TCN OOF is slightly optimistic.** E4 chose its epoch count on the same OOF BCE.
- **The val predictions are built two different ways.** The TCN's val probabilities come from a full refit (3 seeds). The GBDT val probabilities are fold-bagged.
- **Val has 67 evaluable falls and 4 subjects.** Every val CI spans about ±0.04 to ±0.06. **No voting scheme is significantly better than another on val.** The only robust conclusions are the ones that hold on both: radar-agreement voting and stacking hurt, and simple averaging is the safe default.
- **No subject-disjoint (SR-CV) folds and no pseudo-test bank were used.** These were not available in the 60-minute box. The selection should be re-checked under X4's SR-CV rule before promotion.
- **Hyperparameters were not tuned.** Each GBDT used one sensible configuration. Seed bagging was 2 seeds x 4 folds, not the 5 seeds X5 asks for.
- **Boundaries were not touched.** The median start error is 0.24-0.34 s. The TCN-based blends have the worse tie-break 1 (0.34 s), so they should be paired with E2's annotator replay for boundaries.

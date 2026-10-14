# E4: Deep temporal model (TCN) over 0.2 s block features

## What this is
This is a dilated residual **TCN** in PyTorch on CPU. It reads the whole per-0.2 s block feature sequence of a stream and outputs a framewise fall probability.

- **Inputs, 112 channels.**
  - The 64 prototype "full" features, taken from `analysis/prototype_detector.py`: static change, frame-difference energy, jump, structure features and the train-only PCA fingerprint.
  - 48 range-band channels per radar, built from the cached `mot_map` and `prof_02`. Each is the log mean over 8 bin bands (0-4, 5-12, 13-21, 22-39, 40-52, 53-68, 69-99, 100-183) and is normalised by the stream median.
- **Model.** 1x1 input conv, then 7 residual blocks (kernel 3, dilations 1..64, 64 channels, GELU, dropout 0.2), then a 1x1 output. The receptive field is 255 blocks, about 51 s, non-causal. Input channel dropout is 0.1 and Gaussian input noise is 0.05.
- **Training.** Plain BCE (flagged falls stay positives, as in the prototype). AdamW at lr 1e-3. Random 200 s windows, batch 16, 20 steps per "epoch". The code supports a 2-stage MS-TCN (`--stages 2`), but it was **not run**. No start/end heads were built.
- **Post-processing.** The prototype's `to_events` and its 60-point grid (thr, smooth, min_dur, gap), tuned on train OOF only with the official evaluator code.

## Protocol (val touched only for final scores)
1. **CV.** I used the prototype's 4-fold leave-streams-out `GroupKFold` on the 13 train streams. For every fold and seed I logged the held-out BCE after each epoch.
2. **Early stopping.** E\* is the epoch with the lowest held-out BCE averaged over folds and seeds. OOF is the seed-mean of the held-out probabilities at E\*.
3. **Configuration choice, on train OOF only.** With 1-seed CV, the 112-channel input reached OOF rank F1 0.9274 against 0.9225 for the 64 prototype features alone. I picked 112 channels.
4. **Final model.** I re-ran the CV with 3 seeds (`max_epochs 14`), getting E\*=8 and OOF 0.9404. I then fitted 3 seeds on all 13 train streams for 8 epochs and averaged their val probabilities. Val was scored once with the official CLI (`--ignore-flagged`).
5. **Ensemble weight.** The HGB+TCN averaging weight w_tcn ∈ {0.25, 0.5, 0.75} was chosen on train OOF. w=0.75 won, and w=0.5 is reported as the pre-registered default.

The HGB reference is the prototype "full" model, re-run in `hgb_ref.py` with the same features, folds, parameters and grid. It reproduces **train OOF 0.9009 and val 0.8840 exactly**.

Caveat: casting the features to float32 alone moved HGB val to 0.871. That is the prototype's own seed/early-stopping sensitivity (see X5), so the cache keeps float64.

## Results (official evaluator, val, --ignore-flagged)

| model | train OOF rank F1 | **val rank F1** | F1@0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | FA/h @0.3 | median \|start err\| @0.3 | mean tIoU |
|---|---|---|---|---|---|---|---|
| HGB prototype "full" (reference) | 0.9009 | **0.8840** | 0.948/0.919/0.785/0.504 | 62/6/5 | 2.93 | **0.22 s** | 0.698 |
| TCN, 3-seed bag (`tcn_all`) | 0.9404 | **0.9303** | 0.970/0.970/0.851/0.507 | 65/2/2 | 0.98 | 0.34 s | 0.695 |
| 0.75·TCN + 0.25·HGB (w chosen on OOF) | 0.9428 | **0.9343** | 0.985/0.970/0.848/0.561 | 64/1/3 | 0.49 | 0.33 s | 0.704 |
| 0.5·TCN + 0.5·HGB (pre-registered default) | 0.9340 | 0.9235 | 0.963/0.963/0.844/0.548 | 65/3/2 | 1.46 | 0.28 s | 0.706 |

Train OOF is leaky for both models (shared ADL files and subjects across streams), so only relative OOF comparisons are meaningful.

### Uncertainty
- **Paired segment-cluster bootstrap** on val (`bootstrap.py`; 588 composition segments resampled 5000x; official `_match_stream`; point estimates equal the CLI exactly):

  | comparison | Δ | 95% CI | P(Δ>0) |
  |---|---|---|---|
  | TCN − HGB | +0.046 | [−0.001, +0.099] | 0.97 |
  | ens 0.75 − HGB | +0.050 | [+0.007, +0.101] | 0.99 |
  | ens 0.75 − TCN | +0.004 | [−0.022, +0.034] | 0.58 |

- **Seed variance** (`seed_var.py`). These are 5 **single-seed** TCNs with the same config and post-processing. Seeds 1000-1002 are the same ones used in the 3-seed bag. Their val scores are 0.902 / 0.899 / 0.879 / 0.881 / 0.899, so **mean 0.892 ± 0.011**.
  - The 3-seed bag scored 0.930, well above every member. Bagging clearly helps, but 0.930 is a single draw of a 3-seed bag. A realistic expectation for this model is roughly 0.89-0.93, not a guaranteed +0.046.
  - Caveat: the post-processing was tuned on 3-seed-mean OOF probabilities, which are smoother than single-seed ones. This may penalise the single seeds somewhat.

### Does it beat or complement the prototype?
- **It beats it on detection.**
  - Block-level val AUC is 0.981 against 0.962, and AP 0.749 against 0.683. OOF is similar (AUC 0.9936 vs 0.9932, AP 0.859 vs 0.843).
  - Event level: FN @0.3 goes from 5 to 2 and FP from 6 to 2. The 2 TCN misses are a subset of the 5 HGB misses.
  - @0.5 the misses are 10 against 14: 8 are shared, 6 only HGB, 2 only TCN.
- **Errors are strongly correlated.** The Pearson r of the probabilities is 0.90 OOF / 0.88 val. For the residuals (y−p) it is 0.78 OOF / 0.86 val, and 0.81 on fall blocks in val. Per-event best-tIoU correlation on val is 0.76. This is expected, because the TCN consumes the same 64 features plus range bands.
- **Averaging gives only a small, insignificant gain over the TCN alone**: +0.004, CI [−0.022, +0.034]. It mainly lowers FA/h (0.98 to 0.49) and lifts F1@0.7 (0.507 to 0.561).
- **Weakness: boundaries and tie-break 1.** The median start error is 0.33-0.34 s against 0.22 s, and mean tIoU is unchanged. The TCN finds more falls but does not localise them better. It should therefore be combined with the annotator replay boundary model (X1/X2, E2) and not used for boundaries.

## Recommendation (for the team lead)
1. Use the TCN as the **detection / probability member**.
   - Add a 3-5 seed bag. The single seeds vary by ±0.011 and the bag scored better than every single seed.
   - Average it with the HGB (w≈0.5-0.75) mainly to cut FAs.
   - Hand boundaries to the replay/selector (E2).
2. Before promotion, re-check under the subject-disjoint SR-CV / pseudo-test bank (E1, X4), using a ≥5-seed paired comparison. The val CI on TCN − HGB still touches 0.
3. Cheap next steps that were not done here:
   - Feed the variance-normalised junction statistic and the X1 segment-relative replay signals as extra channels.
   - Set sample weight 0 for flagged ±2 s instead of positive labels.
   - Try a 2-stage MS-TCN (implemented, untested).

## Honest limitations
- **The CV is leave-streams-out, not subject/ADL-disjoint.** Train OOF is leaky. E\*, the feature-set choice, the post-processing and w were all selected on it.
- **n = 67 evaluable val falls.** The val gain over HGB is borderline at 95%.
- **Seed noise is of the same order as some of the gaps.** The feature-set choice (+0.005 OOF) was made with 1 seed and is within noise.
- **The model reuses the prototype's features**, including the junction/structure and fingerprint features, which are generator artefacts. It therefore inherits their rule and drift risk. A physics-only TCN was not run.
- **D3 cannot be judged here.** No evaluable D3 falls exist in val.
- **No pseudo-test bank or SR-CV evaluation was done**, because of the time box.
- **Compute.** The 1-seed CV takes about 8.5 min. The 3-seed CV plus the final fit plus val takes 16 min (`wall_s` 974). The 5-seed variance run takes about 4 min. All runs used 2 threads with peak RSS below about 1 GB (observed about 0.85 GB in `ps`).

## How to run
All commands run from this directory with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2`:
```
python3 data.py                       # cache/blocks.npz (prototype features + range bands, folds)   ~1 min
python3 hgb_ref.py                    # HGB reference OOF/val probs; reproduces 0.9009 / 0.8840
python3 tcn.py --tag cv1_proto --feats proto --seeds 1 --max_epochs 25 --steps 20 --no_val   # selection runs
python3 tcn.py --tag cv1_all   --feats all   --seeds 1 --max_epochs 25 --steps 20 --no_val
python3 tcn.py --tag tcn_all   --feats all   --seeds 3 --max_epochs 14 --steps 20           # final (CV + val)
python3 compare.py tcn_all 0.25 0.5 0.75   # correlations, miss overlap, OOF-chosen ensemble weight, val
python3 bootstrap.py out/val_hgb.csv out/val_tcn_all.csv out/val_ens_tcn_all_w0.75.csv out/val_ens_tcn_all_w0.5.csv
python3 seed_var.py                   # single-seed val spread
```

Outputs are in `out/`:
- `res_*.json`: config, BCE curve, E\*, OOF score, post-processing and val summary.
- `probs_*.npz`: OOF and val probabilities.
- `val_*.csv` / `val_*.json`: submissions and official reports.
- `compare_tcn_all.json`, `bootstrap_vs_*.json`, `seed_var.json`.

Logs are in `logs/`. `lib.py` holds the shared post-processing and scoring: the official CLI for val, and the in-process `abc2027.evaluate` for train OOF tuning.

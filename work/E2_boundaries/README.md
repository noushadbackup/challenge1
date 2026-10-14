# E2: boundary refinement (annotator replay on distinct frames)

**Result:** with the prototype's detections unchanged, the val ranking F1 goes from **0.8840 to 0.9284**. F1@0.5 goes from 0.785 to 0.904, F1@0.7 from 0.504 to 0.726, and the median |start error| @0.3 from 0.22 s to 0.06 s. All of these numbers come from the official evaluator with `--ignore-flagged`. The configuration was chosen on train OOF only, and val was scored once per reported candidate (2 candidates).

## What it does

1. **Candidates (`s1_proto.py`).** This reproduces the prototype `full` detector exactly: same features, HGB parameters, `GroupKFold(4)` by stream, and post-processing grid. It gives the train OOF events and the val events. The val score reproduces 0.8840.
2. **Annotator replay (`s2_replay.py`).** The ground truth is produced by `abc2027.annotate.propose_fall`, so each candidate gets that function, run on stream crops read from the raw h5. Only the crop is read, about 1 s per event.
   - **Distinct-frame motion (the main new ingredient, X1 P1).** Bit-exact sample-and-hold repeats are removed per radar before building each `RadarTrack`. The true frame times are kept, so the annotator computes `diff²` motion on distinct frames and interpolates it onto its grid, as it does on raw recordings. On the stream grid, the held frames add zeros and doubled steps to the motion.
   - **37 crop variants**, all on distinct-frame motion except 3 kept as references:
     - `locL_R`: crop `[t_s-L, t_e+R]`.
     - `detL_R`: the same crop, clipped at detected junctions. These use the variance-normalised jump on `prof_02` (q97 peaks, 1 s exclusion gate) and stay 0.2 s inside the junction.
     - `detqL_R`: the same, with q95 peaks and a 0.2 s gate (recall-oriented).
     - `detr*` / `detqr*`: junction refined to frame level from the raw 50 Hz crossfade signature.

     No GT or composition data is used to build the crops.
3. **Selection (`s3_select.py`), train OOF only.** The choice is made in-process with the official scoring code `abc2027.evaluate.evaluate`. Stage 1 searches the crop or crop-set (median of start/end across crops), the agreement gate (tIoU(replay, detector) ≥ g, otherwise the detector bounds are kept) and min_ok. Stage 2 searches the duration floor and end offsets.
   - The frozen configs are then scored with the official CLI on train and on val.
   - **A**: median of the 6 crops `loc2_3d, loc2_5d, det2_3d, det2_5d, detq2_3d, detq2_5d`, gate 0.1.
   - **B** = A plus a 1.0 s duration floor, applied by extending the end only (evaluable GT is never shorter than 1.0 s; this is `MIN_EVENT_S` in annotate.py). The stage-2 grid found **no strict train gain** over A, only ties. B was kept as a structural prior; on train it ties with A on ranking and is +0.02 on F1@0.7.
4. **Diagnostics and paired tests.** `s4_diag.py` gives per-type hit rates, signed errors and the crop-start versus true-junction error. `s5_paired.py` gives paired bootstraps.

## Results (official evaluator CLI, `--ignore-flagged`)

| split | variant | rank F1 | F1@0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | med \|start err\| @0.3 | mean tIoU | FA/h |
|---|---|---|---|---|---|---|---|
| train OOF | prototype | 0.9009 | 0.941/0.919/0.843/0.638 | 170/14/16 | 0.15 | 0.764 | 2.26 |
| train OOF | A replay | 0.9225 | 0.941/0.935/0.892/0.730 | 173/11/13 | 0.04 | 0.830 | 1.78 |
| train OOF | B replay + floor | 0.9225 | 0.941/0.935/0.892/0.751 | 173/11/13 | 0.04 | 0.836 | 1.78 |
| **val** | prototype | 0.8840 | 0.948/0.919/0.785/0.504 | 62/6/5 | 0.22 | 0.698 | 2.93 |
| **val** | A replay | 0.9235 | 0.948/0.933/0.889/0.711 | 63/5/4 | 0.06 | 0.826 | 2.44 |
| **val** | **B replay + floor** | **0.9284** | 0.948/0.933/0.904/0.726 | 63/5/4 | 0.06 | 0.830 | 2.44 |

**Paired B vs prototype.**

| split | stream-cluster bootstrap Δ (90% CI) | P(Δ>0) | event bootstrap 95% CI (FP fixed, optimistic) |
|---|---|---|---|
| train (13 streams) | +0.021 [+0.002, +0.042] | 0.96 | [+0.009, +0.035] |
| val (5 streams) | +0.043 [+0.013, +0.079] | 0.99 | [+0.027, +0.066] |

On val @0.5, 8 events are gained and 0 lost, and FPs drop from 15 to 7. On train @0.5, 15 are gained and 6 lost.

**Per fall type, best-IoU hit@0.5 / @0.7 (prototype → B).**

| type | val | train |
|---|---|---|
| Chute01 | 0.86/0.57 → 0.93/0.86 | 0.83/0.67 → 0.93/0.81 |
| Chute02 | 1.00/0.58 → 1.00/0.74 | 0.87/0.64 → 0.90/0.73 |
| Chute03 | 0.80/0.70 → 1.00/0.85 | 0.90/0.67 → 0.90/0.77 |
| Chute04 | 0.43/0.07 → 0.64/0.43 | 0.73/0.56 → 0.80/0.68 |

Chute04 is still the bottleneck.

**Key ablation (train OOF, ranking F1, gate 0.1).** Distinct-frame motion against stream-grid motion:

| crop | stream grid | distinct frames |
|---|---|---|
| `det2_5` | 0.8865 | 0.9178 |
| `loc2_5` | 0.8883 | 0.9160 |
| `det2_3` | 0.8847 | 0.9171 |

That is about **+0.03 on every crop tested** (full grid in `out/select_stage1_train.csv`). For reference, X2's stream-grid replay reached 0.9086 on val.

Other train findings:
- Long crops hurt (`det10_*`, `loc3_*`), and short crops (L 1.5-2 s) are best.
- Frame-level junction refinement and the q95 junctions bring no gain on their own.
- An earlier version (`out/v1`) tried start shifts. `ds=-0.1` gave +0.009 train rank but moved the median start error from 0.02 to 0.10 s. It was rejected because of tie-break 1.

**Junction diagnostic.** This compares the detected crop start (q97, 1 s gate) with the true clean start of the fall segment. On train 68% are within ±0.3 s; in 31% the detector returned an earlier junction. On val the figures are 85% and 14%. Frame refinement narrows the p95 of the error from 0.17 s to 0.05 s when the junction is found. With q95 and a 0.2 s gate, 86.5% of train candidates are within ±0.3 s (`out/jdiag.py`). The L=2 s cap makes the replay robust to missed junctions, which is why `loc` and `det` crops score about the same.

## How to run

```
cd /home/user/challenge1/work/E2_boundaries && ./run.sh      # ~12 min, n_jobs=2
```

Individual steps:
- `python3 s3_select.py` reuses `out/replay_*.csv` (~20 s).
- `python3 s4_diag.py val out/val_B_replay_post.csv B`
- `python3 s5_paired.py val out/val_proto_cli.csv out/val_B_replay_post.csv`

The submission-format val CSV for B is `out/val_B_replay_post.csv`, with official JSON `out/val_B_replay_post.json`. The chosen configs are in `out/summary.json`.

## Honest limitations

- **Train OOF is leaky for detection** (stream-grouped CV, shared ADL files and subjects). The boundary step has only a few discrete knobs (crop set, gate, floor), but these were selected from about 300 stage-1 configs on 186 evaluable train falls. The train gain of the selected config is therefore mildly optimistic. Many neighbouring configs score 0.915-0.922 on train, so the gain does not hinge on the exact pick. Val was scored only for the two frozen candidates.
- **The selection was not done under subject-disjoint CV** (SR-CV, X3/X4), and only one seed was used. The HGB `early_stopping='auto'` randomness noted by X5 is fixed by `random_state=0` here.
- **Val has 67 evaluable falls from 4 subjects and no evaluable D3.** The D3 / P3_3 behaviour of the replay is not measured.
- **Detection is unchanged.** The gate only replaces bounds; it never adds or removes events. The TP gain @0.3 comes from better boundaries.
- **The replay needs raw h5 crops at inference.** The cost is about 1 s per candidate for 37 variants; the final config needs only 6 (not timed separately). Peak RAM was not measured precisely, but each replay holds one crop of about 15 s × 3 × 368 floats.
- **Dedup relies on bit-exact sample-and-hold repeats in the CIR.** This is a sensor and recording property that the annotator naturally sees on raw files. If the test stream writer changed it (for example by interpolating instead of holding), this part of the gain would shrink. That risk is not measured; the stream-grid replay (about +0.02) is the fallback.
- **Not done (next steps):**
  - LightGBM selector or residual over replay internals (review reasons, spread across crops, anchor).
  - Crop-spread as a confidence feature for the hit model.
  - Re-validating on SR-CV folds and on held-out P3_3.

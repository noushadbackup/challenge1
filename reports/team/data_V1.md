# V1: Data engineer (data integrity and file formats)

Scratch code (all of it run in this session): `/home/user/data_raw/agents/V1/`

| script | purpose |
|---|---|
| `f1_iq_check.py`, `f1_sweep.py` | F1: I/Q pairing tests on raw `segments/*.npy` (no cached features) |
| `f2_check.py` | F2: timing statistics recomputed from the raw CSVs with my own join, plus a signal check that the CSV junctions are real |
| `f10_check.py` | F10: rescored with the official evaluator (in-process) and with my own matcher, a 5-min-window bootstrap, and a 4-fold stream rotation that retrains the lead's feature set |
| `new_checks.py`, `new_checks2.py`, `n2_dup_auc.py`, `n2_fall_window.py`, `n2_control.py` | New findings N1 to N3 |

## How I checked F1, F2 and F10

**F1 (I/Q layout).** I did not use the lead's method (range-time break and correlation of motion profiles). I tested a physical property instead. If channel `a` is I and channel `b` is Q of the same bin, then the complex signal `z = x[a] + j*x[b]` (band-passed to 0.3 to 8 Hz) has a **one-sided Doppler spectrum** when someone walks toward or away from the radar. Two unrelated real channels give a symmetric spectrum. I swept the offset `d` between the paired values on Marcher (walking) source files, using the 8 most active bins per radar, with 72 to 168 pairs per offset.

| offset d | 176 | 180 | 182 | 183 | **184** | 185 | 186 | 190 | 1 | 46 | 92 | 138 | interleaved 2k/2k+1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Doppler asymmetry (median) | .089 | .192 | .347 | .424 | **.451** | .411 | .308 | .192 | .145 | .098 | .047 | .029 | .143 |

The asymmetry peaks sharply at exactly d = 184. The values at d = 183 and 185 stay high only because neighbouring range bins are correlated. The quadrature test agrees: |corr(Q, Hilbert(I))| is 0.254 at d = 184, against 0.216 and 0.235 at d = 183 and 185, and 0.06 at d = 1. The I/Q variance ratio is also closest to 1 at d = 184 (|log ratio| 0.146, against 0.30 at d = 183). The envelope sqrt(I² + Q²) is smoother along range than |I| alone (roughness 0.364 against 0.458), and an interleaved layout is much rougher (0.650). That smoothness test alone cannot tell 183 from 184 (0.357 against 0.364). **Conclusion: F1 holds.**

**F2 (timing structure).** I joined `event_id` to `segment_id` myself and asserted that every event lies in a fall segment with the same source path.

| quantity | result |
|---|---|
| fall start minus segment start | 0 to 7.86 s, median 3.03 s |
| segment end minus fall end | 3.26 to 10.0 s, median 7.56 s |
| fall segment length | 7.32 to 18.64 s |
| fall duration | 0.48 to 5.0 s, median 1.74 s |
| Autre segments | 77 to 120 s |
| Empty segments | 82 to 120 s |
| fall density | 58.1 per hour in both train and val |
| fall-to-fall joins | 82 |

Signal check: on the cached `prof_02`, the static-profile jump in a ±1 s window around each CSV junction has a median of 1.09, against 0.22 at random times (AUC 0.855). The argmax falls within ±0.4 s of the CSV time for 95 % of junctions, so the CSV junction times match the signal.

Small corrections to the report:
- The minimum gap between falls, measured from the end of one fall to the start of the next, is **5.9 s**. The report gives 7.4 s, which may use a different definition.
- Not every junction has the 0.2 s crossfade. **24 junctions are hard cuts** (overlap 0) and 2 are partial (0.12 and 0.14 s). Every hard cut leads into a fall clip (see N1).

**F10 (prototype at 0.884).**
- I rescored `reports/prototype/val_full.csv` in-process with `abc2027.evaluate` and got **0.884** (0.948 / 0.919 / 0.785 / 0.504). My own independent matcher, written from the README rules, gives the same result: 0.884, with TP/FP/FN at θ = 0.3 of 62/6/5.
- A bootstrap over 26 windows of 5 minutes gives a 95 % CI of **[0.827, 0.940]**.
- Leakage audit: the feature columns contain no `t`, `y` or label-derived fields. The PCA is fitted on train only, and the post-processing comes from train OOF. I found no leak.
- **Rotation (the main caveat).** I retrained the same "full" model with the same post-processing (0.4, 3, 0.8, 0.6) on 4 different held-out stream sets, with val included in training where it was not held out:

  | held-out set | rank F1 | FA/h |
  |---|---|---|
  | train fold A | 0.865 | 0.9 |
  | train fold B | 0.865 | 4.6 |
  | train fold C | 0.832 | 3.6 |
  | val | 0.884 | 2.9 |

  The mean is **0.861**. Val is the best of the 4 folds, so expect about 0.86 rather than 0.88. F1 at θ = 0.7 varies widely across folds, from 0.48 to 0.66.

## Votes

| # | vote | conf | by code | evidence / comment |
|---|---|---|---|---|
| F1 | agree | 0.95 | yes | Doppler asymmetry peaks at d = 184 (.451), against .145 at d = 1, .047 at d = 92 and .143 interleaved. The quadrature and variance-ratio tests agree. |
| F2 | agree | 0.95 | yes | Every number reproduced from the raw CSVs, and the CSV junctions line up with signal jumps. Corrections: the end-to-start minimum gap is 5.9 s, and there are 24 hard-cut junctions. |
| F3 | agree | 0.7 | no | Follows from EMA mechanics and is consistent with my junction jump (median 1.09 against 0.22 at random times). The 84× and 90 % figures were not recomputed. |
| F4 | partially | 0.6 | partly | My simpler detector (±1 s window maximum of the 0.2 s jump) reaches only AUC 0.855. The 0.97 depends on the method and its negative sampling. Plausible, but not reproduced. |
| F5 | agree | 0.6 | no | Plausible from the geometry. Not checked. |
| F6 | agree | 0.65 | no | Consistent with the README ("bounds placed from the change in the static CIR profile"). The AUCs were not recomputed. |
| F7 | partially | 0.6 | no | "Fall recording vs Autre/Empty is about chance" holds only for the amplitude fingerprint. Duplicate-frame rate separates fall clips from Autre/Empty at segment level with AUC 0.89 on train and on val (N2). |
| F8 | agree | 0.9 | yes | Flagged counts from the CSV: val D3 39/39, train D3 104/120 (87 %), evaluable val D1 33 and D2 34 (67 in total). Also note that 26 to 32 % of train D1/D2 falls are flagged too (31/120 and 38/119). |
| F9 | agree | 0.7 | no | Expected from the shuffling planner in `streams.py`. Not recomputed. |
| F10 | agree | 0.85 | yes | 0.884 reproduced exactly by two scorers. CI [0.83, 0.94]. The rotation mean is 0.861, and val is the most favourable fold. |

## New findings

**N1: A fall that starts at the junction is always an ignore zone, and every hard cut leads into a fall clip.**
- 27 falls (21 train, 6 val) have `source_t_start` < 0.25 s, so the fall starts at the junction (24 of them at exactly 0).
- **All 27 have `needs_review = True`, so 0 are evaluable.** By dataset: D2 13, D3 12, D1 2.
- The planner sets crossfade = 0 when the event starts at the start of the clip. That is why there are 24 hard cuts (overlap 0), and every one of them leads into a fall clip.

Implication: a fall onset that coincides with a junction never needs a prediction, and a prediction there cannot become a false alarm, because the ignore zone is widened by ±2 s. Two things follow. Training should not teach the model "junction jump = fall onset" from these 27 cases: down-weight or drop them as positives. The hard-cut signature (no 0.2 s crossfade) also marks a "free" zone at test time.

**N2: About 25 % of the 50 Hz frames are exact duplicates. Their rate depends on the corpus and drops during the fall.**
- `align.py` resamples by nearest frame. Frames that share an integer Unix second are spread over equal slots, so seconds with fewer than 50 frames repeat frames.
- In the raw segments, 19 to 24 % of frames are exact copies of the previous frame. Most duplicates occur on all 3 radars at once (0.18 of frames).
- Duplicate rate by segment type: ADL 0.255, fall clips 0.236, Autre/Empty 0.274. The implied rate of distinct frames is about 37.5 per second.
- **Inside the annotated fall, the duplicate rate falls to 0.189.** A random window of the same length elsewhere in the same clip gives 0.265 (n = 477, Wilcoxon p = 1.6e-16, AUC 0.67). The effect appears in D1, D2 and D3.

This is an acquisition or annotation artifact that is immune to junctions and to amplitude. It is a cheap weak feature: duplicate fraction per 0.2 or 1 s block. It also separates fall clips from Autre/Empty (segment AUC 0.89).

Engineering consequence: frame-difference energy (`e_diff`) contains 25 % structural zeros, and any Doppler, phase-velocity or STFT feature computed on the 50 Hz grid assumes uniform sampling, which is false. To fix this, compute differences between distinct frames (drop duplicates, or divide by the real time step), and add the duplicate fraction as an explicit feature.

**N3: Label durations have a spike at exactly 2.00 s.**
- 35 of 478 events last exactly 2.00 s. The next most common duration, 1.06 s, has 12 events.
- These 35 events are 24 D2 and 14 Chute04; 27 of them are evaluable.
- **11 of the 67 evaluable val falls (16 %) are exactly 2.00 s.**
- 96 % of the source start and end times lie on the 0.02 s grid.

Implication for boundary regression (θ = 0.5 and 0.7): the distribution of durations is not smooth. A "default 2 s" mode exists, probably manual or default bounds, especially for Chute04. Consider a mixture prior on duration, or snapping the predicted duration to 2.0 s when the regressor is uncertain, and check its effect on θ = 0.5 and 0.7 with train OOF.

# Expert review X4: competition strategy and ML validation

Reviewer: X4 (CV design, shake-up risk, ensembling, metric optimisation within the rules).
Scope: PLAN.md v1, REPORT.md, DATA_TEAM_VERDICT.md, the official `evaluate.py` / `streams.py` / `build_streams.py`, and the X1/X2 reviews.

**Verdict: sound with changes.** The pipeline is right. The **validation protocol (PLAN §2) is the weakest part of the plan**: it cannot see the main source of test variance, and its pseudo-test bank tunes on validation subjects.

All numbers below come from scripts I ran in this session, in `/home/user/data_raw/agents/X4/`:

- `x4_shakeup.py` (+ `.log`): subject-cluster bootstrap and test-like subject draws.
- `x4_errors.py`: where the false negatives and false positives come from.
- `x4_segfill.py`: segment-completion upper bound.
- `x4_pad.py`: padding tuned on train OOF.
- `x4_mindur.py`: duration floor.

Every score uses the official matcher (`abc2027.evaluate`, `--ignore-flagged` semantics), either in-process or through the CLI. "Replay" means X2's train-selected configuration (`x2_*_det2_3_gate0.3.csv`). "Train OOF" means the prototype's leave-streams-out OOF, which is **leaky** (verdict N5).

---

## 1. What I measured

### 1a. Test variance comes from subjects, not from streams or seeds

The val set has **4 evaluable fall subjects** (P1_8, P1_9, P2_8, P2_9). Across all splits there are only 16 D1/D2 subjects, plus P3_3. The test is very likely the same shape: about 2 D1 + 2 D2 new subjects.

Per-subject ranking F1 on the leaky train OOF (prototype) runs from **0.51 (P2_5) to 0.97 (P3_3)**.

| quantity (official matcher) | prototype | replay (X2) |
|---|---|---|
| val ranking F1 | 0.8840 | 0.9086 |
| val subject-cluster bootstrap, 95% CI | [0.818, 0.934], sd 0.030 | – |
| val paired gain, replay − prototype (cluster bootstrap) | – | +0.024, CI [+0.005, +0.044], P(>0) = 0.99 |
| train-OOF ranking F1 | 0.7900 | 0.8179 |
| train-OOF paired gain | – | +0.028, CI [−0.001, +0.057], P(>0) = 0.97 |
| **test-like draw**: 2 D1 + 2 D2 subjects drawn from the 8+8 train+val subjects, 4000 draws | **mean 0.799, sd 0.051, p5–p95 [0.708, 0.875]** | gain +0.034, sd 0.020; P(replay wins on the drawn test) = 0.96 |

Consequences:

- **A fixed model's test score moves by about ±0.05 (1 sd) depending only on which 4 subjects the organisers drew.** Val 0.884 is one such draw, and it sits above the subject-draw mean. The agreed "planning baseline ≈ 0.86" is optimistic; the honest range is about 0.80–0.88. *(Estimate: built from leaky OOF outcomes of a weaker fold model.)*
- **Every team gets the same subjects, so the ranking depends on paired differences.** Those are about 2.5x less noisy (sd 0.020 vs 0.051). Model selection must therefore be **paired and subject-clustered**, not "mean ± sd over regenerated streams".
- **The PLAN §2 pseudo-test bank (≥ 50 streams regenerated from the *validation* segments) re-shuffles the same 119 falls from the same 4 subjects.** Its sd measures context and margin noise only, and underestimates the real uncertainty by roughly 2x. It also turns val into a tuning set, which contradicts the verdict ("do not tune on val").

### 1b. Where the errors are

From `x4_errors.py`:

- Of the prototype's val predictions, **84 of 85 lie inside fall-recording segments**, and only 1 FP is in Autre. **There are 0 FPs in ADL** on val, and **0 FPs outside fall segments** on train OOF.
- The detector already separates fall recordings from ADL. The remaining errors are *inside* fall segments: the wrong burst or the wrong boundary.
- Val at θ0.5 (prototype): FN = 3 with no overlapping prediction + 11 boundary misses. FP = 14 inside fall segments + 1 elsewhere. The replay cuts the boundary misses to 4 and the in-segment FPs to 8.
- Train OOF: **43 of 186 evaluable falls have no overlapping prediction at all** (121 fall segments get no prediction). Recall at segment level is the bigger lever on train subjects. On val it is small (3 such FNs).
- Predictions per hour are 41.5 (val) and 39.2 (train OOF), against the generator's fixed density of **58.08 fall segments per hour**. The density is the same in every split, because `finish_plan` uses one binding density.

### 1c. Cheap post-processing (tuned on train OOF, applied once to val)

| post-processing on top of the replay | train OOF | val | median start error |
|---|---|---|---|
| none | 0.8179 | 0.9086 | 0.04 / 0.06 s |
| best (Δstart, Δend) from a train grid: (−0.3, +0.3) | 0.8299 | 0.9086 (+0.000) | **0.28 s** (tie-break 1 lost) |
| duration floor 1.2 s, extending `t_end` (evaluable GT minimum is 1.00 s) | 0.8239 | **0.9136** | 0.04 / 0.06 s (unchanged) |
| same floor on the prototype | 0.790 → 0.808 | 0.884 → 0.889 | unchanged |

The val padding surface is flat for Δend ≥ 0 and Δstart ∈ [−0.4, 0]. It falls steeply for Δstart > 0.

- **After the replay, start padding buys nothing and costs tie-break 1.**
- A pure-end floor is free in both splits.

### 1d. Segment completion: one prediction per fall-recording segment (upper bound)

Setup: oracle segment identity from the composition file. For every fall segment with no prediction, I added one event at the frame-difference peak (start = peak − 0.7 s, length 1.74 s).

| | added | before | after | effect |
|---|---|---|---|---|
| train OOF | 121 | 0.790 | **0.831–0.836** | +0.04 |
| val | 37 | 0.909 | 0.895 | −0.013: the val misses are wrong-burst errors inside segments that already have a prediction, and 2 FPs were added |

The evidence is mixed and the segments are oracle, so this is P1, to be decided by the new CV.

---

## 2. Changes (prioritised)

### P0 (must do to win)

**P0-1. Replace PLAN §2 with subject-disjoint regenerated CV (SR-CV).**

Folds. Use 4 folds, each shaped like the test:

- 2 D1 + 2 D2 + 2 D3 fall subjects, together with their own Autre/Empty files, which live under each fall subject's folder;
- plus 2 ADL subjects (the ADL subject is the trailing number of the file: train {4, 5, 6, 9, 10, 12}, val {0, 11}).
- **Fold 4 is the official val**, and its official streams count as one seed. Folds 1–3 split the 6/6/6 train fall subjects and the 6 train ADL subjects.
- Keep P3_3, the only evaluable D3 subject, alone with its fold, so that D3 generalisation is measured honestly.

Streams:

- **Held-out streams.** Build them with `abc2027.streams.prepare_split(split="test", …)` + `finish_plan(bundle, 58.08)` + `build_streams._write_stream`. This gives `adl_uses=2`, `ignore = flagged` and the official crossfade. Use 3 seeds per fold.
- **Training streams.** Build them from the other 3 folds with `split="train"` (`adl_uses=3`). Use ≥ 2 seeds; the extra seeds act as legal re-crop augmentation from package data.
- **Disk.** Write one h5 at a time, run `extract_features` on it, then delete it. About 400 MB is peak per stream; the cost is roughly 8 min of feature extraction per seed for all folds *(estimate from REPORT: 2 min per 8.2 h)*.

Use of the folds:

- All hyperparameters, post-processing, the replay crop/gate (X2) and the thresholds are chosen on **pooled SR-CV OOF**: about 250 evaluable falls from about 20 subjects.
- The official val is no longer a sacred holdout: with 4 subjects it carries too little information. It is reported as fold 4.
- The pseudo-test bank built from val segments is dropped as a selection tool.

Why: it removes the verdict's leak (ADL files and subjects shared across train streams). It is also the only design whose variance matches the test (§1a).

Expected gain: none directly. It prevents choosing the wrong model when the paired true gain is below about 0.03; on 4 subjects that happens often (§1a: P(win) is only 0.96 even for a +0.034 change).

Verify:

- SR-CV fold 4 reproduces the official val score for the same model within ±0.005 (sanity check of the regenerated pipeline).
- Rerun the prototype in SR-CV and report the per-fold and per-subject table.

**P0-2. Promotion rule: replace "pseudo-test mean > 1 sd".** A change is kept only if all of the following hold:

1. The paired subject-cluster bootstrap on pooled SR-CV OOF gives P(Δranking F1 > 0) ≥ 0.90. Resample fall subjects; FPs go with the segment they fall in (code: `x4_shakeup.py::boot`).
2. At least 3 of the 4 folds are non-negative.
3. The worst-subject ranking F1 does not drop by more than 0.02.
4. The median start error at θ0.3 does not rise by more than 0.05 s, unless Δranking F1 ≥ 0.01.

Keep a log of every comparison made, to count garden-of-forking-paths decisions.

**P0-3. Choose the operating point for the metric, not for probability 0.5.**

- The prediction set is hard: the score only orders duplicates, and X2 showed it has no effect.
- For micro-F1, adding a candidate helps exactly when its hit probability exceeds F*/2. See Lipton, Elkan & Narayanaswamy, *Thresholding classifiers to maximize F1*, https://arxiv.org/abs/1402.1892 (cited from memory, not re-fetched).
- With mean-over-θ ranking: include a candidate iff mean over θ ∈ {0.1, 0.3, 0.5} of P(tIoU ≥ θ) > F̄/2 ≈ 0.43–0.45.
- Implementation:
  - Train an event-level hit model (LightGBM) on SR-CV OOF candidates. Its targets are three binary labels, hit@0.1/0.3/0.5.
  - Features: detector peak probability, replay agreement (tIoU with the detector), multi-crop spread, segment-classifier probability and the quality-gate replica.
  - Threshold its mean output at F̄/2, re-estimated on OOF.
  - Report the F1-vs-threshold curve, and require a plateau of at least ±0.05 in threshold.
- Final model: the **fold ensemble** (the average of the 4 SR-CV models, each trained on 3/4 of the data), so the OOF threshold transfers without recalibration.
  - If a full refit on train+val is used instead, map the threshold through matched quantiles of the OOF score distribution. Do not reuse the raw value.

**P0-4. Protect tie-break 1 and use the structural constraints for free.**

- **Start padding = 0.** Measured: −0.3 s gives +0.000 val and lifts the median start error from 0.06 s to 0.28 s.
- **Duration floor of 1.2 s, extending `t_end` only.** Measured: +0.005 val, +0.006 train OOF, start error unchanged.
- Clip `t_start` to ≥ J_prev + 0.2 s, because GT never crosses a junction. This one is untested.
- Tune Δend ≥ 0 on SR-CV.

**P0-5. Submission safety and test-drift checks, the day test is released.**

- `validate_predictions()` raises on any bad row and rejects the whole file. Run the official `validate_predictions` against `streams/test`, with ids `test_00..04`, t_start < t_end, times clipped to `[0, duration]` and no NaN, in CI before every upload.
- **Adversarial validation:**
  - Train a LightGBM classifier at segment level to tell train+val from test, using junction rate per hour, segment-length distribution, crossfade signature, repeat rate and spacing-3 share, fingerprint PCA and the per-radar noise floor.
  - An AUC above about 0.6 means the hidden `rebuild_test.py` differs. In that case, switch to the physics-only hedge configuration that was frozen in advance.
  - The README states the test is written by `scripts/rebuild_test.py`, which is not shipped, so drift is a real possibility.
- **Count check.** Predicted events per hour on test should match SR-CV OOF (about 40/h; 58.08/h is the hard ceiling). A deviation above 15% signals a calibration shift.

### P1

**P1-1. Segment-level recall floor (decide in SR-CV).**

- Add a segment classifier (fall recording vs ADL vs Autre/Empty), using junction detection, then the fingerprint, the spacing-3 share and the static change (verdict N1, N3).
- Emit one candidate in every fall-recording segment that has none, localised by the replay. It passes through the P0-3 hit model like any other candidate.
- Measured upper bound: +0.04 train OOF, −0.013 val (§1d). Expected gain +0.00 to +0.03, depending on the subjects.
- This matches X2 P1-1. I would raise it to P0 if SR-CV confirms the gain on 3 of 4 folds.

**P1-2. Ensembling.**

- Members: 3–5 diverse members (LightGBM with 2 feature views, XGBoost or CatBoost, one TCN or CRNN). Bag 3 seeds each.
- Fusion: an **equal-weight mean (or rank mean) of framewise probabilities**, then one decoder, then the replay for boundaries.
- Do **not** fit a logistic stacking meta-model or learned weights: with about 250 evaluable events this overfits. Allow weights only if they pass P0-2.
- No event-level WBF and no soft-NMS (X2).
- Expected gain: +0.005 to +0.015 (estimate).

**P1-3. Physics-only hedge as a switch, not a permanent member.**

- Measure its cost in SR-CV (REPORT: 0.790 vs 0.884 on val).
- Keep it in the blend only if the cost is ≤ 0.005. Otherwise ship it as the fallback that P0-5's adversarial check triggers.

**P1-4. Training labels.**

- Mask flagged events ±2 s out of the loss (sample weight 0); do not label them negative.
- Drop the hard-cut and junction-start falls (verdict N6) as positives.
- At inference, emit flagged-looking candidates anyway: unmatched predictions within flagged ±2 s are free. This is X2 P0-4; use X1's quality-gate replica as the feature.

**P1-5. D3.** P3_3-type subjects score 0.97 on train OOF. A test D3 subject with ~20% flags could add about 15 evaluable falls, so never down-weight radar-0 far-bin features. Measure D3 only through the P3_3 fold.

### P2

**P2-1. Transductive ADL duplicate mask (verdict N4): do not use it.** Zero val ADL FPs and zero non-fall-segment FPs on train OOF means about 0 gain for a real rule-interpretation risk. Revisit only if a future model produces ADL FPs in SR-CV.

**P2-2. Bayes-optimal interval decoding.**

- Run X2's multi-crop replay, then pick the (s, e) that maximises the mean over θ of P(tIoU ≥ θ) under the empirical crop spread.
- Expected gain: ≤ +0.01 (estimate). The surface in §1c is already flat.

**P2-3. Isotonic calibration only to make the P0-3 threshold portable.** It has no ranking value.

---

## 3. Plan text to amend

- **§2.** Replace points 1–5 with P0-1 and P0-2.
- **§1 table.** Replace "pseudo-test distribution ≥ 50 regenerated val streams" with "SR-CV per-fold and per-subject ranking F1, paired cluster bootstrap".
- **§3 S5.** Remove "isotonic calibration (greedy matching uses the score order)". Add "event hit model + F̄/2 rule".
- **§4 exit criteria.** Use SR-CV numbers, not absolute val numbers. For example, P2 exit: pooled SR-CV ranking F1 ≥ prototype + 0.03 with P(>0) ≥ 0.9, and worst fold ≥ 0.80.
- **§5 risks.** Add "test subject draw: sd ≈ 0.05" and "hidden rebuild_test.py drift → adversarial validation + hedge switch".

## 4. Metrics to add (every run, per fold)

- Ranking F1 per fold and **per subject**, with the paired cluster-bootstrap CI against the current best.
- Worst-subject and worst-fold ranking F1.
- The error decomposition from §1b:
  - FN with no overlapping prediction vs FN from a boundary miss;
  - FP inside a fall segment (wrong burst) vs FP in ADL/Autre/Empty vs FP within ±2 s of a flagged event (free);
  - fall segments with 0, 1 or ≥2 predictions.
- Predicted events per hour vs SR-CV reference vs the 58.08/h ceiling.
- F1-vs-threshold curve and plateau width; mean-θ hit-model reliability.
- Median start error at θ0.3 (tie-break 1), always next to ranking F1.
- At test release: adversarial-validation AUC, junction rate per hour, and segment-length KS statistic vs train/val.

## 5. Risks

- **Subject lottery.** About 4 evaluable test subjects means ±0.05 around the expected score. It cannot be removed, only reduced, by fixing the worst subjects (P2_5, P2_1 and P1_5 are the hard train subjects).
- **The hidden `rebuild_test.py` may differ from `build_streams.py`.** Junction, fingerprint and replay cues could break. Mitigation: P0-5.
- **SR-CV fidelity.** Regenerated folds must reproduce official val (P0-1 check). With only one evaluable D3 subject, the D3 estimate rests on n = 1.
- **Forking paths.** Many post-processing knobs against about 250 OOF events. Mitigations: the log in P0-2, plateau checks, and freezing settings before looking at val/fold 4.
- **The numbers in §1a and §1c rest partly on leaky train OOF** (verdict N5) and on X2's replay outputs. The test-draw sd is an estimate.

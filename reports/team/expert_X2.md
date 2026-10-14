# Expert review X2: temporal action localisation and sound-event detection

Reviewer: X2 (ActionFormer, TriDet, MS-TCN, CRNN, boundary regression, event post-processing).
Scope: PLAN.md v1, REPORT.md, the data team's verdict, `analysis/prototype_detector.py`, and the official `abc2027.evaluate` and `abc2027.annotate`.
Scripts and outputs are in `/home/user/data_raw/agents/X2/` (`replay_on_preds.py`, `replay_sweep.py`, `paired.py`, `sweep_{val,train}.log`, `x2_*.csv/json`).
Every number below comes from those scripts, run in this session and scored with the official evaluator using `--ignore-flagged`, unless it is marked as an estimate.

## Verdict: sound, with changes to stages S3 to S5

The segment-centred plan is right: junctions, then segment context, then localisation inside the segment.
From a TAL point of view, three parts of the plan are aimed at the wrong target.

1. **Boundaries come from a known algorithm, so replay it instead of learning it.** The ground truth is produced by `abc2027.annotate.propose_fall`, which ships in the package (verdict N7). The best boundary model is that algorithm run on the stream crop around each detection. Learned regressors (the ruptures plus LightGBM offset model, or the ActionFormer and TriDet heads) can only approximate it, and they would learn from about 250 evaluable falls.
2. **The metric is F1 at fixed tIoU, not mAP.** The usual TAL toolkit tunes for AP: dense proposals, soft-NMS and score calibration. Here a duplicate proposal is a false positive, and the order of the scores does not matter. I checked that second point (table below).
3. **Deep TAL models (S3-b, S3-c, S3-e) belong in P2, not on the critical path.** Detection at θ0.1/0.3 is already about 0.95. What remains is in the boundaries, and those are handled by item 1.

## Evidence from this session

### E1. Replaying the annotator on the stream crop of each prototype detection

Setup:
- Start from the prototype's val predictions (`reports/prototype/val_full.csv`).
- For each detection, read the raw h5 slice and run `propose_fall` on a crop `[max(J_prev+0.2, s−L), min(J_next−0.2, e+R)]`.
- `J` is either the oracle junctions or a simple detected static-profile jump. The detector uses the prof_02 jump, divides it by a 10 s rolling median, keeps the top 3%, and ignores jumps within ±1 s of the detection.
- "gate 0.3" means the replay is accepted only if it has tIoU ≥ 0.3 with the prototype's own interval. Otherwise the prototype interval is kept.
- **The configuration was selected on train OOF predictions** (`tmp_train.csv`, the leaky OOF; the replay itself has no learned parameters) and then applied to val. The val sweep was also run, and it is shown for transparency.

| val (official evaluator) | ranking F1 | F1@0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | median start err | mean tIoU | FA/h |
|---|---|---|---|---|---|---|
| prototype (REPORT §7) | 0.8840 | .948/.919/.785/.504 | 62/6/5 | 0.22 s | 0.698 | 2.93 |
| **replay, detected junctions, L=2, R=3, gate 0.3 (train-selected)** | **0.9086** | .933/.919/**.874/.667** | 62/6/5 | **0.06 s** | 0.810 | 2.93 |
| replay, oracle junctions (crop = true segment, cap 10/12 s), no gate | 0.9118 | .941/.926/.868/.721 | 63/6/4 | 0.04 s | 0.822 | 2.93 |
| replay, local window only L=4, R=5, no gate (bad window) | 0.6906 | .748/.705/.619/.446 | 49/23/18 | 0.14 s | 0.728 | 11.2 |

On train OOF the same selected configuration gives 0.790 → 0.818, with median start error 0.28 → 0.04 s.

On val, the paired per-event comparison at θ0.5 shows 7 events gained and 1 lost. The bootstrap gain is +0.025, 95% CI [+0.008, +0.042]. This CI resamples events only and holds the FP count fixed, so the true CI is wider.

The gain does not come only from tie-break 1: median start error falls from 0.22 to 0.06 s, but F1@0.5 also rises by 0.09 and F1@0.7 by 0.16.

**The crop window is critical.** A crop that reaches past the true junction gives the annotator the wrong "standing" profile (`_profiles` takes the first 1 s and the last 1.5 s of the crop). It then locks onto the junction and turns true positives into false positives (row 4). This is why the gate and an accurate junction stage matter.

Per-type hit rate at tIoU 0.5 / 0.7 on val (best-overlap prediction):

| fall type | prototype | replay, detected window L=2 | replay, oracle junction window |
|---|---|---|---|
| Chute01 (n=14) | .86/.57 | .86/.71 | .71/.64 |
| Chute02 (n=19) | 1.00/.58 | 1.00/.68 | 1.00/.68 |
| Chute03 (n=20) | .80/.70 | 1.00/.85 | .95/.85 |
| **Chute04 (n=14)** | **.43/.07** | .57/.36 | **.79/.71** |

Chute04 is the boundary bottleneck. With the true crop start (the junction) it reaches 0.79/0.71, but the fixed 2 s pre-window cuts it short. The fix is a better junction-anchored window, not a better regressor.

### E2. The order of the scores has no effect on the ranking

I re-scored both submissions with the original scores, with all scores set to 1, and with the scores reversed. All three give the same result: 0.8840 and 0.9086.

The reason is that predictions do not overlap, and falls are at least 5.9 s apart, so greedy-by-score matching never has a choice to make. **Isotonic calibration "because greedy matching uses the score order" (S5) therefore has zero value.** Only the decision threshold matters. If the score is used at all, it is only for the event-level threshold.

## Changes to the plan (prioritised)

### P0: must do to win

**P0-1. Replace S4 with "annotator replay plus a learned fallback".**
- For each candidate segment, run `abc2027.annotate.propose_fall` on the stream crop `[J_prev+0.2 s, min(J_next−0.2, e+R)]`. Fall back to `[s−2, e+3]` when no confident junction is found within 8 s before the detection.
- Accept the replay when it agrees with the detector (tIoU ≥ g). Otherwise keep the detector's interval.
- Tune `g`, `R` and the junction-confidence threshold on stream-grouped train CV and the pseudo-test bank, never on val.
- Optional second step: a LightGBM residual model on (replay start/end minus GT). Features: replay `t_peak`, `t_settle`, `onset_spread_s`, `single_radar_share`, `review` reasons, fall-type proxies and the duration prior (2.00 s spike).
- Drop `ruptures` PELT as the main boundary model. Keep it only as a hedge feature.
- Expected gain: +0.025 measured with a simple detector. Up to +0.028 or more with oracle-quality junctions, and Chute04 hit@0.7 rising from 0.36 to about 0.7 (oracle, measured).

**P0-2. Make the junction stage serve the crop, not just the features.**
- Use the variance-normalised junction statistic (verdict N3) and pick its threshold for **recall on into-fall junctions** (L2's table: q0.95 gives 0.95 recall).
- Exclude jumps that fall inside the detected fall: the fall itself is a static change.
- Metric: |detected crop start − true junction| for fall segments, plus the share within ±0.3 s.
- Expected gain: closes the measured gap between detected and oracle crops: 0.909 → 0.912 overall, and mostly Chute04 at θ0.5/0.7.

**P0-3. Post-processing for F1, not AP.**
- Use hard 1-D NMS. Allow at most **one event per junction-to-junction segment**: the generator puts exactly one fall in each fall crop (478 fall segments for 478 events, see `numbers.json`).
- Do not use soft-NMS. Do not keep multiple proposals per event (each duplicate is a false positive).
- Choose the decision threshold by maximising the *mean of F1 over θ0.1/0.3/0.5* on grouped CV, and report the F1-versus-threshold curve. A flat plateau is required before freezing.
- Remove "isotonic calibration for greedy matching" from S5 (E2 shows it has no effect).
- Expected gain: robustness more than points; it prevents duplicate-FP regressions when the ensemble is added.

**P0-4. Ignore zones: use them only when they can be predicted.**
- Following L2's correction, a prediction is free only if it overlaps a flagged event ±2 s.
- `propose_fall` returns `review` and `quality` on the crop. Train a "will this be flagged" head: start not visible, under 1 s, radar onsets disagree, low residual.
- When the head is confident, emit the prediction anyway: it costs nothing if the event is flagged and gains a TP if it is not. Never drop detections because they look like D3.
- Expected gain: estimated 0 to +0.01 (not measured).

### P1

**P1-1. Reframe detection as segment classification.**
- After junctions are found, each segment is "fall crop with one fall", "ADL" or "Autre/Empty".
- Train a segment classifier (LightGBM). Features: maximum static change, peak frame-difference burst, fingerprint, spacing-3 repeat share, segment length.
- Combine it with the frame model as `P(event) = P(fall segment) × P(burst at t | fall segment)`.
- This is the SED "weak-label (clip) plus strong-label (frame)" structure from DCASE Task 4: clip-level gating of frame posteriors.
- Expected gain: fewer FPs on Autre/Empty and pre-fall walking; estimated +0.01, unverified.

**P1-2. Boundary metrics on every run**, in addition to PLAN §1:
- tIoU histogram and signed start and end errors per fall type and per dataset;
- hit@0.5 and hit@0.7 per type;
- replay-vs-detector agreement rate;
- share of TPs whose crop start lies within 0.3 s of the true junction;
- F1@0.7 as an early warning for boundary quality, even though it is not ranked.

**P1-3. Multi-window replay ensemble.**
- Run the replay on 3 to 5 crops: the junction crop, L=1.5/2/2.5, and R=3/5.
- Take the median `t_start` and `t_end` across crops that pass the gate. The spread across crops is a confidence feature for the residual model.
- Cheap: the replay of 85 val detections takes about 10 s on 1 CPU.

### P2: only if P0 and P1 are saturated

**P2-1. Deep TAL members as detection diversity, not as boundary models.**
- Use MS-TCN++ (https://github.com/sj-li/MS-TCN2, MIT) or a small CRNN (Conv over range bins, BiGRU over time, DCASE-baseline style) on the 0.2 s feature sequence.
- Output framewise P(fall), fused by averaging *probabilities* before event extraction (not WBF of events).
- ActionFormer (https://github.com/happyharrycn/actionformer_release, MIT) and TriDet (https://github.com/dingfengshi/TriDet) are AP-tuned and data-hungry: about 250 evaluable falls against thousands in THUMOS. If they are tried, use TemporalMaxer (https://github.com/TuanTNG/TemporalMaxer), which is lighter, and use only their classification output.
- Keep the replay for boundaries.
- Promotion rule: the member must add at least 1 pseudo-test sd in ensemble.
- Expected gain: 0 to +0.01 (estimate). (The repo URLs are cited from memory and were not re-fetched this session.)

**P2-2. Event-level evaluation library for sanity checks.**
- `sed_eval` (https://github.com/TUT-ARG/sed_eval) and `sed_scores_eval` (https://github.com/fgnt/sed_scores_eval) for threshold-independent PSDS-style curves across operating points.
- The official evaluator stays the source of truth.

## Validation requirements (TAL-specific)

1. Tune the replay crop and gate on **stream-grouped, subject-grouped train CV and the pseudo-test bank**.
   - The sweep here shows that the window choice swings val by 0.22 (0.69 to 0.91). That makes it a dangerous hyperparameter to set on 67 val falls.
   - The config selected on train (L=2, R=3, gate 0.3) also ranks among the best on val, which is reassuring.
2. Re-run E1 on the pseudo-test bank, which has known junctions, and report the mean ± sd of (replay − prototype).
3. Re-run E1 once the junction detector changes. The replay depends on junctions in a non-linear way (row 4 above).
4. The test uses the same generator and annotator. If a later package version changed `annotate.py`, the replay advantage would shrink to the physics of the "static step plus motion burst" definition. Keep the learned fallback (P0-1) as the hedge.

## Risks

- Gate and window were selected on the leaky train OOF; the val gain is +0.025 with a lower CI bound of +0.008 (event bootstrap, FP fixed, so optimistic).
- If junctions are missed, the crop includes the previous ADL segment. The annotator then places the fall at the junction, which turns TPs into FPs (worst case measured: 0.69).
- Stream crops have a 0.2 s crossfade, and the source "after" profile comes from the recording end, which the crop lacks. A residual gap to GT therefore remains even with oracle junctions: θ0.7 F1 is 0.72, not 1.0.
- Using the organiser's annotator code at inference is allowed by the stated rules (package code, open-source), but the team lead should confirm this together with the other generator-artefact cues.

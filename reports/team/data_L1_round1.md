# L1 (lead, justifier) round 1: verdicts on F1-F10 and the analysts' new findings

All L1 numbers below come from scripts I ran in this session, in `/home/user/data_raw/agents/L1/`:
`a_csv.py` (CSV-only checks), `b_feats.py` + `b_an.py` (cached features, all 18 streams),
`d_f4f7.py` (junction detector and fingerprint), `e_boot.py` (official `_match_stream` with a cluster bootstrap),
`c_hash.py` (exact frame hashing on raw val h5), and official-evaluator runs on `reports/prototype/val_full.csv` and two shifted copies of it.
When a number comes only from an analyst, I name the analyst.

## Summary table

| ID | Verdict | One-line corrected statement |
|---|---|---|
| F1 | confirmed | I = 0..183, Q = 184..367. Three analysts (V1, V2, V6) ran independent offset scans, and each peaks at exactly d=184. |
| F2 | confirmed (minor correction) | All margins reproduce. The minimum end-to-start gap between falls is 5.9 s; 7.4 s is the start-to-start gap. Evaluable falls have pre-margin >= 1.0 s. |
| F3 | corrected | The junction spike is about 55x the ADL median (p10-p90 12-118x), not 84x. It decays below 3x in about 5 s (median), not about 8 s. The 90%-of-FAs-within-1 s figure holds. |
| F4 | corrected | Block AUC 0.97 for the plain jump is confirmed. The event P/R for the plain jump is about 0.86/0.70, not 0.83/0.82. A variance-normalised statistic gives P 0.92-0.97, R 0.78-0.87. |
| F5 | corrected | D1 maps to R29444 and D2 to R30722 (confirmed). D3 is NOT weak on all radars: it sits on R29171 (radar 0) at far bins of about 40-63. |
| F6 | corrected | Static-change AUC is about 0.98 for D1 and D2. For D3 it is 0.60 over all falls (not 0.76) and 0.96 over the 16 evaluable falls. The "motion peak ~1.5 s before onset" claim is rejected: it is an EMA/junction artifact. |
| F7 | corrected | The fingerprint is stronger than claimed against ADL (val AUC 0.94-0.98). Against Autre/Empty it is about 0.70, not chance (only 6 val segments). |
| F8 | confirmed (nuance) | The counts are exact. Flags follow deterministic annotator rules. D3 flagging depends on the subject (P3_3 is 20% flagged). |
| F9 | confirmed | Two independent permutation tests give p ~0.4-0.5, and the generator does a single shuffle. |
| F10 | confirmed (caveats) | 0.884 reproduced exactly. The cluster-bootstrap 95% CI is [0.826, 0.940]. Train OOF is leaky, and rotation folds suggest an expected score of about 0.86-0.88. |

## F1 - I/Q layout: CONFIRMED
- V1 (Doppler asymmetry sweep, raw Marcher segments), V2 (offset scan o=-4..4, raw val_00/train_03) and V6 (improperness scan, 25 raw falls) each ran their own code. All three put the optimum at exactly d=184; the neighbours d=183/185 are clearly worse, and interleaved pairing or d=1 are near noise.
- V2 also shows that far-bin I and Q have equal std and are uncorrelated (|r| <= 0.03). That is what true quadrature looks like.
- Three different test statistics agree, and there are no dissenting votes. I did not re-run this.

## F2 - stream construction and timing: CONFIRMED, minor corrections
- `a_csv.py` reproduces the following: pre-margin 0.00-7.86 s (median 3.03); post-margin 3.26-10.0 s (median 7.56); fall segment 7.32-18.64 s (median 12.40); 58.08 falls/h. Every one of the 478 events lies inside exactly one fall segment.
- Correction: the minimum gap between consecutive falls is **5.90 s end-to-start**. The report's 7.4 s is the start-to-start minimum.
- Nuance (V3, V4, V7, and confirmed by L1): evaluable falls have pre-margin >= 1.00 s. This follows directly from `annotate.py` `ONSET_VISIBLE_S = 1.0` (the start_not_visible flag), because about 85% of fall crops start at source t=0. The onset prior for evaluable falls is therefore [junction + 1 s, junction + 7.9 s].

## F3 - EMA junction artifact: CORRECTED (magnitude), mechanism confirmed
- `b_feats.py` takes the peak of the radar-summed cached e_mot in [j-0.5, j+3] s, over all 2328 junctions, and divides it by the median e_mot in ADL at least 5 s from any junction. The result is a median of **55.5x** (p10 12.2x, p90 117.7x). By junction type: ADL->ADL 44x, ADL->fall 77x, fall->ADL 74x, fall->fall 94x, transitions involving Autre/Empty 69-109x.
- Two analysts got the same scale: V2 54x (p10-p90 11-112x) and V4 63.5x, the latter with the official `subtract_background` on raw h5. The 84x in numbers.json uses a definition that none of us can reproduce. Report about 55-65x and give the definition with it.
- Decay to below 3x the reference (0.5 s smoothing) takes a median of **4.8 s** (p90 9.3 s). V2 and V4 both found 5.1 s (p90 9.8-10.6). So the "about 8 s" in the finding is closer to a p85-p90 than to a typical value.
- 90% of energy-baseline FAs start within 1 s of a junction: reproduced independently by V2 (rebuilt baseline, official evaluator: 0.3151, TP 34 / FP 100) and by V4 (official `_match_stream`). Both get 90.0%, with a median offset of +0.14 s. I did not re-run this part, but two independent code paths agree exactly.

## F4 - junction detectability: CORRECTED
- `d_f4f7.py` uses the complex prof_02 with 1 s windows on each side and a 0.2 s guard. Plain relative jump: val block AUC **0.969**, which confirms "about 0.97" (V2 0.972). With the threshold tuned on train for event F1 and +-1.5 s matching, val gives **P 0.858 / R 0.696**, almost exactly V2's 0.86/0.71. The finding's R 0.82 / P 0.83 is not reproduced.
- Variance-normalised statistic: val event **P 0.970 / R 0.775** (block AUC 0.954 in my version). The analysts got P 0.971 / R 0.873 / AUC 0.993 (V2) and P 0.922 / R 0.811 / AUC 0.974 (V5). The precision gain over the plain jump is robust. The size of the recall gain depends on the implementation.
- Missed junctions are mostly ADL->ADL (recall 0.52-0.72). Junctions bounding fall clips are found at 0.89-1.00 (my numbers and V2/V5). Those are the junctions that matter for the onset prior.
- V1's 0.855 comes from a different statistic and different negative sampling, so it does not contradict this.

## F5 - radar per mat: CORRECTED
- `b_feats.py` takes the max over each event of the complex prof_02 block-difference divided by the per-radar, per-bin ADL median. Dominant radar:
  - D1: R29444 for 157/160 (122/122 unflagged); peak bin median 13 (IQR 9-21).
  - D2: R30722 for 159/159; bin 11 (IQR 9-14).
  - **D3: R29171 (radar 0) for 152/159 (16/16 unflagged)**; peak bin median 59 (IQR 43-62).
- V3, V6 and V8 report the same mapping with three different statistics. "D3 weak on all radars" is wrong. D3 is weaker than D1/D2 by roughly 15x vs 200-500x ADL (V3/V6/V8), but it is clearly localised on radar 0.
- The disagreement over the D3 bin (V3 ~43, V6 ~60, V8 ~62) is explained by subject: the peak bins are bimodal. P3_3/P3_6/P3_8 sit at about 61-62, and P3_1/P3_2/P3_5/P3_9 at about 44-47. Features for D3 should cover R29171 bins of about 38-65.
- "Low bins 10-13" holds for D1/D2 only.

## F6 - fall vs static change and motion timing: CORRECTED
- `b_feats.py` compares median |I+jQ| profiles at [t0-2, t0-1] and [t1+1, t1+2], relative-norm change, against 1523 random ADL windows. AUC: **D1 0.982** (evaluable 0.979), **D2 0.987** (0.984), **D3 0.595 over all falls, 0.964 over the 16 evaluable falls**.
- Matches: V3 got D3 0.64/0.63, V7 0.55 (all) and 0.977 (evaluable). The report's 0.76 for D3 is not reproduced by anyone. The headline should be "~0.98 for every evaluable fall type".
- "Motion peak usually ~1.5 s before onset": **rejected as a fall property.** Using radar-summed e_diff smoothed over 0.2 s, the segment max lies inside the evaluable fall for **90% (D1) and 87% (D2)** of cases. The median is **+0.74 s / +1.06 s** after onset, and 87%/82% fall in [0, 2] s. With e_mot (the EMA residual) only 18%/40% are inside the fall, because the junction spike wins. V3 and V7 report the same thing (89-93% inside, +0.73 to +1.10 s). The 11%/35% figures do not appear in any script (V3, V7).
- Caveat: e_diff is not fully junction-immune. The 0.2 s crossfade also produces frame differences. For D3 (weak), the summed e_diff peak lands inside the fall in only 6% of cases, and its median is -1.2 s, at the junction. Use per-radar, mat-cell-restricted motion.
- The last sentence of F6 ("e_diff peaks 0-2 s after onset for D1/D2") is confirmed.

## F7 - static amplitude fingerprint: CORRECTED
- `d_f4f7.py` uses log amp_1s with blocks at least 1.5 s from junctions and falls, StandardScaler + PCA30 + logistic regression, trained on train and scored on val:
  - fall-recording margins vs ADL: val block AUC **0.943**;
  - fall-recording vs Autre/Empty: **0.696**.
- Independent results: V4 0.956 / 0.69-0.70, V6 0.980 / 0.74. The ADL separation is therefore stronger than the 0.91-0.92 in the finding. Against Autre/Empty it is about 0.70, not chance (0.56). Val has only 6 Autre/Empty segments, so that number is noisy.
- Practical weight: V5's paired bootstrap shows that adding the fingerprint to the prototype gives +0.008, 95% CI [-0.051, +0.065], which is not significant. ADL rejection is already solved by other features.

## F8 - flagged falls: CONFIRMED with nuance
- `a_csv.py` crosstab: train D1 31/120 flagged, D2 38/119, D3 104/120 (86.7%); val D1 7/40, D2 6/40, D3 39/39. Evaluable val = 33 + 34 + 0 = **67**. The `ignore` column is false for all 478 rows, so `--ignore-flagged` is mandatory (V4, V7).
- The flags follow `annotate.py` rules: quality gate 5.12e-6, and `ONSET_VISIBLE_S`, `MIN_EVENT_S` and `FLOOR_VISIBLE_S` = 1.0 s. Results: 141/159 D3 falls are quality=low, D1/D2 flags are all needs_review, and the minimum evaluable source onset and duration are both exactly 1.00 s.
- Nuance (V6, confirmed): D3 flagging depends on the subject. 7 subjects are 100% flagged and Test_P3_3 is 20% flagged; it supplies all 16 evaluable train D3 falls. If a test D3 subject records like P3_3, its falls will count. Keep D3 detection alive (radar 0, far bins).

## F9 - no activity context: CONFIRMED
- V5: within-stream permutation test, p=0.485. V7: TV distance permutation, p=0.40. Both match the lead's chi2 p=0.494.
- Structurally, `finish_plan` does one shuffle (V4, V7). I did not re-run this. Two independent code checks plus the generator code are sufficient.

## F10 - prototype score: CONFIRMED with caveats
- The official evaluator on `reports/prototype/val_full.csv` gives ranking **0.8840** (F1 0.948 / 0.919 / 0.785 / 0.504 at theta 0.1 / 0.3 / 0.5 / 0.7), TP 62 / FP 6 / FN 5 at 0.3, and a median start error of 0.22 s.
- `e_boot.py` (official `_match_stream`, 26 five-minute windows, 2000 resamples) gives a 95% CI of **[0.826, 0.940]**, the same as V1 and V5.
- No val labels are used (V3, V4, V8). However, the train OOF used for tuning is leaky. See NF6 below. All 90/90 train ADL files appear in more than one train stream, and each train fall subject appears in 9-12 of 13 streams.
- V1's rotation folds give a mean of 0.861, and val is the most favourable fold. V8's LSO over 18 streams gives 0.880. Plan with an expected score of about 0.86-0.89, not 0.884 as a point estimate.

## Analysts' new findings - L1 verdicts

### Kept (confirmed by L1 code)
1. **NF1. Sample-and-hold duplicate frames, with a corpus- and fall-dependent rate** (V1 N2, V2, V4, V8).
   - Measured with cached e_diff: 23-25% of frames are exact repeats on all three radars at once, and 25-28% on any single radar.
   - Segment rates: fall clips 0.214, ADL 0.245, Autre 0.250, Empty 0.249.
   - Segment AUC for fall clip vs Autre/Empty is **0.78 on train and 0.81 on val** (V1/V4 report 0.89-0.90 with a raw-frame definition). Fall clip vs ADL is only 0.58-0.62.
   - Inside annotated falls the rate is **0.172**, against 0.218 elsewhere in the same clip (Wilcoxon p=1.2e-10). This holds in D1, D2 and D3.
   - Action: compute differences between distinct frames, and add the per-block duplicate fraction as a feature.
2. **NF2. ADL content is reused exactly within a split; fall and Autre/Empty content never is** (V7, V8; V4 for the composition side).
   - Composition: 90/90 train ADL files and 30/30 val ADL files appear in more than one stream. There is no ADL file or subject overlap between train and val.
   - Exact-frame hashing on raw val (`c_hash.py`, blake2b of each 3x368 frame; a match counts if it is in another stream or more than 2 s away in the same stream): **97.3% of ADL frames** are duplicated, against **0.0%** of fall-clip and Autre/Empty frames. 0/463 ADL segments fall below 50% duplicated, while 119/119 fall segments and 6/6 Autre/Empty segments do. This matches V8 (97.3%) and V7 (97%).
   - Rules caveat: using this on test is transductive and relies on a generator artifact. The team lead must decide. Keep a detector that does not depend on it.
3. **NF3. Junction-start falls and hard cuts are all ignore zones** (V1 N1, V4).
   - All 24 zero-crossfade junctions lead into fall clips, and all 24 of those falls have pre-margin < 0.05 s and are flagged.
   - 27/27 falls with source_t_start < 0.25 s are needs_review (train 21, val 6; D2 13, D3 12, D1 2).
   - The evaluator (`IGNORE_MARGIN_S = 2.0`) does not count an unmatched prediction that overlaps an ignore zone widened by +-2 s as an FP.
   - Action: drop these falls as training positives (or down-weight them). After a hard cut, use a permissive threshold.
4. **NF4. Duration spike at exactly 2.00 s** (V1 N3).
   - 35/478 events last exactly 2.00 s; the next most common duration, 1.06 s, has 12. Of the 35, 27 are evaluable, 24 are D2 and 14 are Chute04. 11 of the 67 evaluable val falls are exactly 2.00 s.
   - All evaluable falls last at least 1.00 s (44/478 overall last less than 1.0 s). Enforce dur >= 1.0 s in post-processing.
5. **NF5. GT is produced by the package's own `annotate.propose_fall`** (V7).
   - Verified by code: the start/end come from `_transition_bounds` on the static-profile state and motion. The flags come from the constants above and reproduce the observed flag pattern exactly (minimum onset and duration 1.00 s; D3 quality=low).
   - V7's replay accuracy (tIoU >= 0.5 for 86.6% of 253 evaluable falls on the stream crop): I re-summarised it from V7's `replay_crop.csv` but did not re-run the replay. It is a strong lead for the theta=0.5 term.
6. **NF6. Train OOF / stream-grouped CV leaks** (V4, V8).
   - Confirmed from the composition and events: every train ADL file is in more than one train stream, and each train subject is in 9-12 of 13 streams. Val shares nothing with train.
   - Action: group folds by subject and ADL file, and keep val as the untouched holdout.
7. **NF7. Interval padding tuned on train transfers to val, but the gain is not significant** (V5, V8, V4).
   - Official evaluator: shift (-0.3 s, +0.2 s) gives **0.904** (F1@0.5 0.830, but F1@0.7 0.311 and median start error 0.40 s). Shift (-0.1, +0.2) gives **0.894**.
   - Paired cluster bootstrap of the difference: [-0.005, +0.046], P(<=0)=0.08; and [-0.009, +0.030], P(<=0)=0.23.
   - Adopt padding as a hyperparameter tuned on leak-free OOF. Watch the start-error tie-break.
8. **NF8. Chute04 is the boundary bottleneck** (V6, V8). Prototype best-IoU on val, by type: Chute04 (n=14) hit@0.3 0.71, @0.5 **0.43**, @0.7 **0.07**; Chute01-03 0.93-1.00, 0.80-1.00, 0.57-0.70.
9. **NF9. Val cannot resolve differences below about 0.05** (V5, V1, V8). The CI [0.826, 0.940] is confirmed, and seeds alone move the score 0.871-0.884 (V5). Select models on pooled leak-free CV with paired bootstrap.
10. **NF10. D3 lives on R29171 far bins, and its flagging depends on the subject** (V6, V3, V8). Confirmed. See F5 and F8.
11. **NF11. The variance-normalised junction statistic raises event precision** (V2, V5).
    - Confirmed: P 0.86 -> 0.97 on val in my implementation, with recall 0.70 -> 0.78.
    - The block-AUC gain (0.97 -> 0.99) was **not** reproduced by me (0.969 -> 0.954), so the gain is at the event level.

### Not kept as confirmed (plausible, not verified by L1)
- V3 "annotated start is physical, end is not" (end +0.7-1.1 s before motion stops). This is consistent with propose_fall using state settling, but I did not re-run it.
- V3 "static change across a junction beats falls". This is plausible (junction spikes of 55x) and implies that static change must be computed within detected segments. Not re-run.
- V6 "subject signal strength varies 10x; val subjects are easy". Not re-run.
- V8 "val LSO 0.928 vs 0.884". Not re-run.
- V2 "Doppler sign / phase velocity usable for t_end". F1 supports that the data allow it, but its usefulness is untested.
- V7 "public seed does not reproduce composition". Not re-run.

## Appendix - exact-frame hashing (c_hash.py), raw val h5, 5 streams
- Stream lengths in frames: val_00 88189, val_01 88421, val_02 88084, val_03 88526, val_04 15570.
- ADL: frame-weighted far-duplicate fraction 0.973; segments with dup < 0.5: 0/463.
- Fall clips: 0.000; 119/119.
- Autre/Empty: 0.000; 6/6.
